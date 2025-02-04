from jaxpm.pm import pm_forces
from jaxpm.growth import E, growth_factor as Gp, Gf, dGfa, gp
from diffrax import ODETerm
from diffrax._custom_types import RealScalarLike
import jax

def symplectic_fpm_ode(mesh_shape, dt0 ,  paint_absolute_pos=True, halo_size=0, sharding=None):
    def drift(a, vel, args):
        """
        state is a tuple (position, velocities)
        """
        cosmo = args[0]
        # Get the time steps
        t0 = a
        t1 = a + dt0
        # Set the scale factors
        ai = t0
        ac = (t0 * t1) ** 0.5  # Geometric mean of t0 and t1
        af = t1

        drift_contr = (Gp(cosmo, af) - Gp(cosmo, ai)) / gp(cosmo, ac)
        # Computes the update of position (drift)
        dpos = 1 / (ac**3 * E(cosmo, ac)) * vel

        return dpos * (drift_contr / dt0)

    def kick(a, pos, args):
        """
        state is a tuple (position, velocities)
        """
        # Computes the update of velocity (kick)
        cosmo = args[0]
        # Get the time steps
        t0 = a
        t1 = t0 + dt0
        t2 = t1 + dt0
        t0t1 = (t0 * t1) ** 0.5  # Geometric mean of t0 and t1
        t1t2 = (t1 * t2) ** 0.5  # Geometric mean of t1 and t2
        # Set the scale factors
        ac = t1

        forces = (
            pm_forces(
                pos,
                mesh_shape=mesh_shape,
                paint_absolute_pos=paint_absolute_pos,
                halo_size=halo_size,
                sharding=sharding,
            )
            * 1.5
            * cosmo.Omega_m
        )

        # Computes the update of velocity (kick)
        dvel = 1.0 / (ac**2 * E(cosmo, ac)) * forces
        # First kick control factor
        kick_factor_1 = (Gf(cosmo, t1)   - Gf(cosmo, t0t1)) / dGfa(cosmo, t1)
        # Second kick control factor
        kick_factor_2 = (Gf(cosmo, t1t2) - Gf(cosmo, t1))   / dGfa(cosmo, t1)

        return dvel * ((kick_factor_1 + kick_factor_2) / dt0)

    def first_kick(a, pos, args):
        """
        state is a tuple (position, velocities)
        """
        # Computes the update of velocity (kick)
        cosmo = args[0]
        # Get the time steps
        t0 = a
        t1 = t0 + dt0
        t0t1 = (t0 * t1) ** 0.5  # Geometric mean of t0 and t1

        forces = (
            pm_forces(
                pos,
                mesh_shape=mesh_shape,
                paint_absolute_pos=paint_absolute_pos,
                halo_size=halo_size,
                sharding=sharding,
            )
            * 1.5
            * cosmo.Omega_m
        )

        # Computes the update of velocity (kick)
        dvel = 1.0 / (a**2 * E(cosmo, a)) * forces
        # First kick control factor
        kick_factor = (Gf(cosmo, t0t1) - Gf(cosmo, t0))   / dGfa(cosmo, t0)

        return dvel * (kick_factor / dt0)

    return drift, kick  , first_kick


def symplectic_ode(mesh_shape, paint_absolute_pos=True, halo_size=0, sharding=None):
    def drift(a, vel, args):
        """
        state is a tuple (position, velocities)
        """
        cosmo = args[0]
        # Computes the update of position (drift)
        dpos = 1 / (a**3 * E(cosmo, a)) * vel

        return dpos

    def kick(a, pos, args):
        """
        state is a tuple (position, velocities)
        """
        # Computes the update of velocity (kick)
        cosmo = args[0]

        forces = (
            pm_forces(
                pos,
                mesh_shape=mesh_shape,
                paint_absolute_pos=paint_absolute_pos,
                halo_size=halo_size,
                sharding=sharding,
            )
            * 1.5
            * cosmo.Omega_m
        )

        # Computes the update of velocity (kick)
        dvel = 1.0 / (a**2 * E(cosmo, a)) * forces


        return dvel

    return drift, kick



class DriftODETerm(ODETerm):
    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> RealScalarLike:
        cosmo = kwargs.get("cosmo", None)
        t0t1 = (t0 * t1) ** 0.5  # Geometric mean of t0 and t1

        if cosmo is None:
            return 0.0

        factor =  (Gp(cosmo, t1) - Gp(cosmo, t0)) / gp(cosmo, t0t1)

        return factor


class DoubleKickODETerm(ODETerm):
    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> RealScalarLike:
        cosmo = kwargs.get("cosmo", None)
        t0t1 = (t0 * t1) ** 0.5  # Geometric mean of t0 and t1

        if cosmo is None:
            return 0.0

        t2 = 2 * t1 - t0  # Next time step t2 for the second kick
        t1t2 = (t1 * t2) ** 0.5  # Intermediate scale factor
        kick_factor_1 = (Gf(cosmo, t1)   - Gf(cosmo, t0t1)) / dGfa(cosmo, t1)
        kick_factor_2 = (Gf(cosmo, t1t2) - Gf(cosmo, t1))   / dGfa(cosmo, t1)

        return kick_factor_1 + kick_factor_2
               

class KickODETerm(ODETerm):
    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> RealScalarLike:
        cosmo = kwargs.get("cosmo", None)
        t0t1 = (t0 * t1) ** 0.5  # Geometric mean of t0 and t1

        if cosmo is None:
            return 0.0

        kick_factor =  (Gf(cosmo, t0t1) - Gf(cosmo, t0))   / dGfa(cosmo, t0)  # fmt: skip

        return kick_factor
               
