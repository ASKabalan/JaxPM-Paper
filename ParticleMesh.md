### Cosmological Equations

---

#### **1. Hubble Parameter ($E(a)$)**

The normalized Hubble parameter $E(a)$ is defined as:
$$
E(a) = \sqrt{\Omega_m a^{-3} + \Omega_k a^{-2} + \Omega_{de} a^{f(a)}},
$$
where:
- $\Omega_m$: Matter density parameter.
- $\Omega_k$: Curvature density parameter.
- $\Omega_{de}$: Dark energy density parameter.
- $f(a)$: Dark energy evolution parameter.

---

#### **2. Linear Growth Factor ($D(a)$)**

##### **Definition:**
The growth factor $D(a)$ quantifies the linear growth of structure:
$$
D(a) = \exp \left( \int_{a_{\text{min}}}^a f(a') \frac{da'}{a'} \right),
$$
where $f(a)$ is the growth rate.

##### **ODE Form:**
The growth factor satisfies the second-order differential equation:
$$
\frac{d^2D}{da^2} + \left( \frac{2}{a} + \frac{1}{H(a)} \frac{dH}{da} \right) \frac{dD}{da} - \frac{3 \Omega_m}{2a^5 H(a)^2} D = 0.
$$

---

#### **3. Second-Order Growth Factor ($D_2(a)$)**

##### **Definition:**
The second-order growth factor $D_2(a)$ accounts for nonlinear effects:
$$
D_2(a) = \text{solution to the nonlinear ODE, normalized such that } D_2(a=1) = 1.
$$

##### **ODE Form:**
The second-order growth factor satisfies:
$$
\frac{d^2D_2}{da^2} + \left( \frac{2}{a} + \frac{1}{H(a)} \frac{dH}{da} \right) \frac{dD_2}{da} - \frac{3 \Omega_m}{2a^5 H(a)^2} D_2 = \frac{3 \Omega_m}{2a^5 H(a)^2} D^2.
$$

---

#### **4. Growth Rates**

##### **Linear Growth Rate ($f(a)$):**
The growth rate is the logarithmic derivative of $D(a)$:
$$
f(a) = \frac{d \ln D}{d \ln a} = \frac{a}{D} \frac{dD}{da}.
$$

##### **Second-Order Growth Rate ($f_2(a)$):**
The second-order growth rate is:
$$
f_2(a) = \frac{d \ln D_2}{d \ln a} = \frac{a}{D_2} \frac{dD_2}{da}.
$$

---

### **FastPM Factors**

#### **5. First-Order FastPM Growth Factor ($G_f(a)$)**
The growth factor in FastPM is:
$$
G_f(a) = D_1'(a) \cdot a^3 \cdot E(a),
$$
where:
- $D_1'(a) = f(a) \cdot D(a) / a$.

---

#### **6. Second-Order FastPM Growth Factor ($G_{f2}(a)$)**
The second-order growth factor in FastPM is:
$$
G_{f2}(a) = D_2'(a) \cdot a^3 \cdot E(a),
$$
where:
- $D_2'(a) = f_2(a) \cdot D_2(a) / a$.

---

#### **7. Derivatives of $G_f(a)$ and $G_{f2}(a)$**

##### **Derivative of $G_f(a)$:**
$$
\frac{dG_f}{da} = D_1''(a) \cdot a^3 \cdot E(a) + D_1'(a) \cdot a^3 \cdot \frac{dE}{da} + 3a^2 \cdot E(a) \cdot D_1'(a).
$$

##### **Derivative of $G_{f2}(a)$:**
$$
\frac{dG_{f2}}{da} = D_2''(a) \cdot a^3 \cdot E(a) + D_2'(a) \cdot a^3 \cdot \frac{dE}{da} + 3a^2 \cdot E(a) \cdot D_2'(a).
$$

---

### **Drift and Kick Factors**

#### **8. Drift Factor**
The drift factor for position update is:
$$
\text{Drift} = \frac{1}{a^3 E(a)}.
$$

For scale factors $[a_i, a_c, a_f]$:
$$
\text{Drift}_{\text{mod}} = \frac{1}{a_c^3 E(a_c)} \cdot \frac{G_p(a_f) - G_p(a_i)}{g_p(a_c)},
$$
where:
- $G_p(a)$: Cumulative growth factor for position.
- $g_p(a) = \frac{dG_p}{da}$.

---

#### **9. Kick Factor**
The kick factor for velocity update is:
$$
\text{Kick} = \frac{1}{a^2 E(a)}.
$$

For scale factors $[a_i, a_c, a_f]$:
$$
\text{Kick}_{\text{mod}} = \frac{1}{a_c^2 E(a_c)} \cdot \frac{G_f(a_f) - G_f(a_i)}{g_f(a_c)},
$$
where:
- $G_f(a)$: Cumulative growth factor for velocity.
- $g_f(a) = \frac{dG_f}{da}$.

--- 

This completes the full mathematical description of the relevant quantities and their roles in cosmological simulations.