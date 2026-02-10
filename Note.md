
# MIT 6.801 2020

https://www.youtube.com/watch?v=3NarS3QpaU0

# Aliasing

## 1. Mathematical Derivation of Aliasing

To understand aliasing formally, we use the model of ideal sampling using the Dirac comb.

### The Sampling Process

Let $x(t)$ be a continuous-time signal and $X(f)$ be its continuous Fourier transform (CFT), where $X(f) = \mathcal{F}\{x(t)\}$.

Sampling $x(t)$ at a uniform interval $T_s$ (sampling rate $f_s = 1/T_s$) can be mathematically modeled as multiplying the continuous signal by a **Dirac comb** (impulse train) $s(t)$:

$$s(t) = \sum_{n=-\infty}^{\infty} \delta(t - nT_s)$$

The sampled signal $x_s(t)$ is:

$$x_s(t) = x(t) \cdot s(t) = x(t) \sum_{n=-\infty}^{\infty} \delta(t - nT_s) = \sum_{n=-\infty}^{\infty} x(nT_s) \delta(t - nT_s)$$

### Frequency Domain Representation

To simplify the spectrum of the sampled signal, we apply the Convolution Theorem: Multiplication in the time domain corresponds to convolution in the frequency domain.

$$X_s(f) = \mathcal{F}\{x_s(t)\} = X(f) * S(f)$$

First, we determine the Fourier transform of the Dirac comb $S(f)$. The Fourier series expansion of the periodic impulse train tells us that its transform is also an impulse train in the frequency domain:

$$S(f) = \mathcal{F}\left\{ \sum_{n=-\infty}^{\infty} \delta(t - nT_s) \right\} = \frac{1}{T_s} \sum_{k=-\infty}^{\infty} \delta(f - k f_s)$$

Now, convolving $X(f)$ with $S(f)$:

$$X_s(f) = X(f) * \left[ \frac{1}{T_s} \sum_{k=-\infty}^{\infty} \delta(f - k f_s) \right]$$

$$X_s(f) = \frac{1}{T_s} \sum_{k=-\infty}^{\infty} X(f - k f_s)$$

### The Aliasing Condition

The equation above shows that the spectrum of the sampled signal $X_s(f)$ consists of periodic **replicas** (or images) of the original spectrum $X(f)$, scaled by $1/T_s$ and shifted by integer multiples of the sampling frequency $f_s$.

Let the original signal $x(t)$ be **band-limited** with maximum frequency $B$ (i.e., $X(f) = 0$ for $|f| > B$).

1.  **Fundamental replica ($k=0$):** Centered at $f=0$, covering $[-B, B]$.
2.  **First replica ($k=1$):** Centered at $f_s$, covering $[f_s - B, f_s + B]$.

**Aliasing** occurs when these replicas overlap. The condition for overlap is:
$$f_s - B < B \implies f_s < 2B$$

If $f_s < 2B$, the high-frequency components of the $k=1$ replica "fold over" into the baseband of the $k=0$ replica. A high frequency $f_{in} > f_s/2$ will appear as an alias frequency:
$$f_{alias} = |f_{in} - N f_s|$$
where $N$ is the closest integer.

**Nyquist-Shannon Sampling Theorem:**
To reconstruct $x(t)$ perfectly from $x_s(t)$, we must be able to isolate the $k=0$ replica using a low-pass filter (reconstruction filter). Non-overlap requires:
$$f_s \ge 2B$$

## 2. Mathematical Derivation of Moiré Patterns

Moiré patterns are a visual manifestation of aliasing in the spatial domain, often occurring when two periodic patterns are superimposed. This is equivalent to the **beat frequency** phenomenon in 1D acoustics.

### Superposition of Gratings

Consider two transmission gratings (layers) with sinusoidal transparency.

**Layer 1:** A vertical grating with spatial frequency $k_1$.
$$T_1(x) = 1 + \cos(k_1 x)$$

**Layer 2:** A vertical grating with a slightly different spatial frequency $k_2$.
$$T_2(x) = 1 + \cos(k_2 x)$$

When these layers are superimposed, the resulting transparency is the product of the individual transparencies (multiplicative Moiré):

$$T_{total}(x) = T_1(x) T_2(x) = [1 + \cos(k_1 x)][1 + \cos(k_2 x)]$$

Expanding this product:
$$T_{total}(x) = 1 + \cos(k_1 x) + \cos(k_2 x) + \cos(k_1 x)\cos(k_2 x)$$

Using the trigonometric identity $\cos A \cos B = \frac{1}{2}[\cos(A-B) + \cos(A+B)]$:

$$T_{total}(x) = 1 + \cos(k_1 x) + \cos(k_2 x) + \frac{1}{2}\cos((k_1 + k_2) x) + \frac{1}{2}\cos((k_1 - k_2) x)$$

### The Beat Frequency (Moiré Term)

The expansion contains four frequency components:
1.  $k_1, k_2$: The original high frequencies of the grids.
2.  $k_1 + k_2$: A very high frequency sum component (usually invisible/unresolvable).
3.  **$k_{moire} = |k_1 - k_2|$**: The **difference frequency**.

If $k_1 \approx k_2$, then $K = |k_1 - k_2|$ is a **low spatial frequency** (long wavelength). This is the macroscopic Moiré pattern observed by the eye.

The wavelength of the Moiré pattern $\lambda_{moire}$ is related to the grid steps $p_1$ and $p_2$:
$$\frac{2\pi}{\lambda_{moire}} = \left| \frac{2\pi}{p_1} - \frac{2\pi}{p_2} \right| \implies \lambda_{moire} = \frac{p_1 p_2}{|p_1 - p_2|}$$

### 2D Rotation Moiré

Consider two identical grids with wave vector $\vec{k}$ rotated by an angle $\alpha$.
- Grid 1: $\cos(\vec{k}_1 \cdot \vec{r})$
- Grid 2: $\cos(\vec{k}_2 \cdot \vec{r})$

Where $|\vec{k}_1| = |\vec{k}_2| = k$.
The Moiré wave vector is $\vec{K} = \vec{k}_1 - \vec{k}_2$.

Using the geometry of an isosceles triangle with angle $\alpha$:
$$|\vec{K}| = 2k \sin(\alpha/2)$$

For small angles ($\alpha \ll 1$), $\sin(\alpha/2) \approx \alpha/2$:
$$K \approx k\alpha$$

The Moiré period $D$ is inversely proportional to $K$:
$$ \frac{2\pi}{D} \approx \frac{2\pi}{p} \alpha \implies D \approx \frac{p}{\alpha} $$

This mathematically explains why slight rotations of screens or fences produce enormous, magnifying Moiré bands: the period $D$ is amplified by the factor $1/\alpha$.
## 3. Dataset Verification: aiMotive Multimodal Dataset

The **aiMotive Multimodal Dataset** is an excellent resource for testing SLAM and verifying coordinate system calculations, similar to KITTI.

### Role in Verification
*   **Ground Truth (GNSS/INS):** Provides high-precision GPS and IMU data. This serves as the absolute reference to check if the SLAM-calculated trajectory (relative motion) aligns with global coordinates.
*   **Multimodal Sensor Extrinsics:** Requires solving transformations between Frame_Camera, Frame_LiDAR, Frame_Base, and Frame_World (ENU/Lat-Lon). Validating these transformations is a core part of the "coordinate system calculation" check.
    $$ P_{world} = T_{GPS \to World} \times T_{Base \to GPS} \times T_{Camera \to Base} \times P_{camera} $$
*   **Not for Routing:** While the dataset contains routing data implies navigation, SLAM is distinct from routing. SLAM provides the *Localization* needed for the routing engine to function.

## 4. Inertial Measurement Unit (IMU)

An **IMU** is a sensor that measures specific force, angular rate, and sometimes magnetic field, allowing for "dead reckoning" navigation without external references.

*   **Components:**
    *   **Accelerometer:** Measures proper acceleration (linear motion + gravity).
    *   **Gyroscope:** Measures angular velocity (yaw, pitch, roll rates).
    *   **Magnetometer:** (Optional) Measures magnetic heading.
*   **Usage:** Fills gaps between slow GPS updates (1-10Hz) or camera frames with high-frequency data (200Hz+). Critical for robust visual-inertial odometry (VIO).

## 5. Alternatives to SLAM for GPS Verification

To verify calculated pseudo-GPS coordinates without building a full SLAM map, simpler algorithms can be used:

1.  **Visual Odometry (VO):**
    *   Calculates relative pose $T_{t \to t+1}$ purely from images.
    *   Good for checking local scale and velocity against GPS velocity, even if long-term position drifts.

2.  **Optical Flow Integration:**
    *   Sums 2D pixel vectors.
    *   If properties like camera height $H$ are known, the integrated flow should mathematically match the physical displacement recorded by GPS.

3.  **Sensor Fusion (EKF - Extended Kalman Filter):**
    *   Fuses Wheel Odometry + IMU.
    *   This provides a continuous "Dead Reckoning" path. If this path diverges significantly from the calculated GPS coordinates, the GPS calculation likely has errors (e.g., wrong geodetic datum or projection).

4.  **Structure from Motion (SfM):**
    *   An offline, batch optimization version of SLAM.
    *   Processes all frames at once (Bundle Adjustment) to create a highly accurate "Ground Truth" trajectory to validate real-time GPS calculations against.
## 6. Comparison (Context: aiMotive)

| Method | Type | Strength | Weakness |
| :--- | :--- | :--- | :--- |
| **Geometric VO (ORB)** | **Visual** | Fast, Precise | **Fails Rain/Night** |
| **Optical Flow** | **Visual** | Smooth | **Drifts Fast** |
| **SfM** | **Offline** | Ground Truth | **Not Real-Time** |
| **Radar-Fusion** | **Fusion** | **All-Weather** | Low Resolution |
| **CNN / Deep VO** | **Learned** | Robust | **Black Box** |



## 6. Comparison Table

| Method | Type | Strength | Weakness |
|---|---|---|---|
| VO (ORB) | Visual | Fast | Fails Rain/Night |
| Optical Flow | Visual | Smooth | Drifts |
| SfM | Offline | Accurate | Not Real-Time |
| Radar-Fusion | Fusion | All-Weather | Low Res |
| Deep VO | Learned | Robust | Black Box |

