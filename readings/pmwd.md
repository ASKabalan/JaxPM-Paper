# Reading PWMD paper 

[![arXiv](https://img.shields.io/badge/astro--ph.IM-arXiv%3A2010.11847-B31B1B.svg)](https://arxiv.org/pdf/2211.09815)


## Introduction 

### **Bullet Points with Notes and Complementary Explanations**

---

#### **1. Limitations of Current Workflows**:
   - Statistical inference relies on **summary statistics** (e.g., the power spectrum) to compress cosmological data for modeling.
   - The **power spectrum** is well-suited for Gaussian-distributed data but struggles to capture information in **non-Gaussian data** (e.g., the nonlinear structure of the Universe).
   - **Data compression risks information loss**, meaning not all information in the original dataset is retained in the reduced summary statistics.

   __Note__: 
   Non-Gaussian features are crucial for understanding complex structures in cosmology (e.g., galaxy clustering, filaments, and voids). Data compression amplifies this problem by further reducing the ability to access nonlinear correlations.

---

#### **2. Advantages of Simulations**:
   - **Iterative simulations** can process raw field-level data without reducing it to summary statistics like the power spectrum. This approach preserves the maximum amount of information.
   - Simulations are highly accurate in predicting structure formation, even in the **nonlinear regime**, which is where most summary statistics fail.
   - Simulations naturally account for:
     - **Cross-correlations** between observables (e.g., the connection between galaxy clustering and weak lensing).
     - **Systematic errors**, since simulations can directly incorporate known uncertainties or imperfections in the data.

   __Note__:  
   - Simulations keeping the maximum amount of data. 
   - The field-level approach used in simulations ensures that you retain the **full statistical richness** of the data. This is particularly useful for studying deviations from Gaussianity in the data.

---

#### **3. Challenges with Simulations (Intractability)**:
   - **Intractable** means that something is so difficult or expensive that it cannot be done in practice, even if it is theoretically possible.
   - For cosmological simulations, the challenge is their **high computational cost**. Running detailed simulations of structure formation requires vast computational resources, and older CPU clusters were simply too slow or inefficient.
   - Large datasets compounded the problem, as simulations needed to handle the field-level information for a massive number of data points.

   __Note__: 
   - GPUs have revolutionized this space because they are designed for massively parallel processing, which is ideal for running these types of simulations. What used to take weeks or months on a CPU cluster can now often be done in days or even hours with GPUs.

---

#### **4. Advancements Enabling New Approaches**:
   - **GPUs** provide the computational power necessary to make simulations feasible, even for field-level analysis.
   - **Automatic Differentiation (AD)** libraries, like those in modern machine learning frameworks, allow for the creation of **differentiable models**. This means:
     You can compute gradients of the model outputs with respect to its parameters. thus enables **gradient-based optimization**, which is faster and more efficient for finding cosmological parameter constraints.

   __Note__:  
   - This is a major shift in cosmological analysis. In the past, parameter estimation relied on computationally intensive sampling techniques (e.g., Markov Chain Monte Carlo). With differentiable models, the analysis can be sped up significantly by using gradients to guide parameter estimation directly.

---

#### **5. Proposed Solution (Differentiable Field-Level Forward Model)**:
   - A **differentiable field-level forward model** combines simulation accuracy with new computational tools:
     - It keeps all the information in the data (field-level analysis).
     - It enables joint estimation of **cosmological parameters** (e.g., the density of dark matter) and the **initial conditions** of the Universe.
   - This approach leverages the strengths of simulations, GPUs, and differentiable modeling to maximize both accuracy and computational efficiency.

   __Note__:  
   - This solution is cutting-edge because it unifies several advances into a single framework. The ability to constrain initial conditions and cosmological parameters simultaneously is particularly exciting for improving our understanding of the Universe’s formation and evolution.


   ### Differentiable Simulations and the Adjoint Method

---

#### **1. Early Differentiable Cosmological Simulations**
   - Examples: **BORG**, **ELUCID**, and **BORG-PM**.
   - These models predate modern automatic differentiation (AD) frameworks and relied on **manual (analytic) differentiation**:
     - Derivatives were computed manually using the chain rule and then implemented in code.
     - This was labor-intensive and error-prone.

   __Note__:  
   - While pioneering, these early models faced scalability challenges because manually deriving derivatives becomes cumbersome for complex simulations.
   - Complementary point: These methods marked the beginning of integrating differentiation into cosmological simulations but were limited in scope compared to modern AD systems.

---

#### **2. Introduction of AD Engines for Simulations**
   - Later codes like **FastPM** and **FlowPM** transitioned to using **automatic differentiation (AD)** tools:
     - AD systems (e.g., TensorFlow or custom libraries like vmad) automatically compute gradients by applying the chain rule to primitive operations.
     - This relieves researchers from manually deriving and implementing gradients.
   - Both analytic differentiation and AD methods:
     - Require **storing all intermediate states** to backpropagate gradients through the simulation's entire history.
     - Suffer from a trade-off between **time resolution** (accuracy) and **memory costs**.

   __Note__:  
   - You can think of AD as a “smart assistant” that automates gradient computation, but it still demands a lot of memory. This limitation affects simulations’ accuracy, particularly on small scales or in dense regions where precision is key (e.g., weak lensing).

---

#### **3. Adjoint Method: A Memory-Efficient Alternative**
   - The **adjoint method** systematically computes gradients for time-dependent problems under constraints:
     - It introduces **adjoint variables** (denoted as λ), which are dual to the model's state variables (z).
     - These variables carry the gradient information of the objective function (\(\mathcal{J}\)) with respect to the model’s state (\(\partial \mathcal{J}/\partial z\)).
   - The method works by:
     - Running the simulation **forward in time**, solving the forward equations.
     - Running **backward in time**, solving a set of adjoint equations (dual to the forward equations) to propagate the gradient information to the input parameters (\(\theta\)).
   - This approach avoids storing all intermediate states by **re-simulating the forward states during the backward pass** (if the dynamics are reversible), thereby reducing memory costs.

   __Note__:  
   - The adjoint method addresses a key limitation of AD-based simulations: memory bottlenecks. By re-simulating forward states on demand, the memory complexity becomes independent of the number of time steps.
   - Complementary point: This method is particularly powerful for large-scale, time-dependent problems, such as cosmological simulations, where memory constraints often limit resolution.

---

#### **4. Discretization in Adjoint Simulations**
   - The adjoint method can be applied to **discrete simulations** by deriving the adjoint equations dual to the discrete forward equations:
     - This is called the **discretize-then-optimize approach**, ensuring gradients are computed along the exact same trajectory as the forward simulation.
   - In contrast, the **optimize-then-discretize approach** computes gradients based on a continuous formulation and then discretizes them, which can lead to errors due to mismatched trajectories.

   __Note__:  
   - The **discretize-then-optimize** approach improves accuracy by ensuring the forward and backward paths align, which is critical for reliable gradient computation in simulations.

---

#### **5. Implementation in Cosmological Simulations**
   - The adjoint method is implemented in the new **pmwd library** (written in JAX):
     - **Memory-efficient**: Space complexity is independent of the number of time steps due to reverse time integration.
     - **Computationally efficient**: Optimized for running on GPUs.
   - pmwd enables accurate and scalable gradient computation for cosmological simulations, overcoming limitations of prior methods.

   __Note__:  
   - The use of JAX, a modern AD framework designed for high-performance computing, ensures compatibility with GPUs and scalability for large datasets. This makes pmwd a cutting-edge tool for cosmological research.

---

### **Key Takeaways**
- Early differentiable cosmological simulations relied on manual differentiation, which was labor-intensive.
- Modern AD systems (e.g., TensorFlow) automated this process but introduced memory challenges due to the need to store intermediate states.
- The adjoint method solves these issues by using reverse time integration to propagate gradients without storing intermediate states, making it more memory-efficient.
- The **pmwd library** implements the adjoint method using JAX, combining computational efficiency with memory scalability.