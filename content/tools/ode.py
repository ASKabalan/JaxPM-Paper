from jaxpm.pm import pm_forces
from jax import numpy as jnp
from jaxpm.growth import E, growth_factor as Gp, Gf, dGfa, gp
import jax_cosmo as jc
from jax import lax
from diffrax import ODETerm
from diffrax._custom_types import RealScalarLike


def symplectic_ode(mesh_shape, paint_absolute_pos=True, halo_size=0, sharding=None):
    def drift(a, vel, args):
        """
        state is a tuple (position, velocities)
        """
        cosmo = args
        # Computes the update of position (drift)
        dpos = 1 / (a**3 * E(cosmo, a)) * vel

        return dpos

    def kick(a, pos, args):
        """
        state is a tuple (position, velocities)
        """
        # Computes the update of velocity (kick)
        cosmo = args

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

    return kick, drift


class LeapFrogODETerm(ODETerm):
    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> RealScalarLike:
        action = kwargs.get("action", "D")  # Action type: 'D' for Drift, 'K' for Kick
        cosmo = kwargs.get("cosmo", None)
        t0t1 = (t0 * t1) ** 0.5  # Geometric mean of t0 and t1

        if cosmo is None:
            return 0.0

        if action == "D":  # Drift case
            return (Gp(cosmo, t1) - Gp(cosmo, t0)) / gp(cosmo, t0t1)

        elif action == "FK":
            return (Gf(cosmo, t0t1) - Gf(cosmo, t0)) / dGfa(cosmo, t0)

        elif action == "K":  # Kick case
            last_kick_cond = kwargs.get(
                "cond", False
            )  # True for last kick, False for double kick

            # Dynamic conditions for double kick or last kick
            def double_kick(t0, t1, t0t1):
                # Two kicks combined
                t2 = 2 * t1 - t0  # Next time step t2 for the second kick
                t1t2 = (t1 * t2) ** 0.5  # Intermediate scale factor
                return (Gf(cosmo, t1)   - Gf(cosmo, t0t1)) / dGfa(cosmo, t1) + (
                        Gf(cosmo, t1t2) - Gf(cosmo, t1))   / dGfa(cosmo, t1)  # fmt: skip

            def single_kick(t0, t1, t0t1):
                # Single kick for the final step
                return (Gf(cosmo, t1) - Gf(cosmo, t0t1)) / dGfa(cosmo, t1)

            return lax.cond(last_kick_cond, single_kick, double_kick, t0, t1, t0t1)

        else:
            raise ValueError(f"Unknown action type: {action}")