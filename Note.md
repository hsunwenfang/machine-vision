
# FUN Projects

https://www.cs.cornell.edu/courses/cs5670/2025sp/projects/



# MIT 6.801 2020

https://www.youtube.com/watch?v=3NarS3QpaU0

# Imaging Pipeline

Full imaging pipeline:

    3D world point  X = (X, Y, Z)
          │
          ▼
    ┌─────────────────────┐
    │  Rigid transform     │   [R | t]     rotation + translation
    │  (extrinsics)        │   3×4         NO distortion
    │                      │               NO perspective (yet)
    └──────────┬──────────┘
               │
               ▼
    Camera coordinates  x_cam = R·X + t = (x_c, y_c, z_c)
               │
               ▼
    ┌─────────────────────┐
    │  Perspective         │   (x_c, y_c, z_c) → (x_c/z_c, y_c/z_c)
    │  division            │
    │                      │   THIS is where perspective happens
    └──────────┬──────────┘
               │
               ▼
    Normalised coordinates  (x_n, y_n) = (x_c/z_c, y_c/z_c)
               │
               ▼
    ┌─────────────────────┐
    │  Radial distortion   │   (x_n, y_n) → (x_d, y_d)
    │  (lens model)        │
    │                      │   THIS is where distortion happens
    └──────────┬──────────┘
               │
               ▼
    Distorted normalised coordinates  (x_d, y_d)
               │
               ▼
    ┌─────────────────────┐
    │  Intrinsic matrix K  │   K = [f_x  0   c_x]
    │                      │       [ 0  f_y  c_y]
    │                      │       [ 0   0    1 ]
    └──────────┬──────────┘
               │
               ▼
    Pixel coordinates  (u, v)



          CAMERA 0                                    CAMERA 1
          ────────                                    ────────

3D point X ──────────────────────┬──────────────────────────── X
                                 │
       ┌─────────────┐           │           ┌─────────────┐
       │ R₀·X + t₀   │           │           │ R₁·X + t₁   │
       │ (extrinsics) │           │           │ (extrinsics) │
       └──────┬──────┘           │           └──────┬──────┘
              │                  │                  │
              ▼                  │                  ▼
       ┌─────────────┐           │           ┌─────────────┐
       │  ÷ z_c      │           │           │  ÷ z_c      │
       │ (perspect.) │           │           │ (perspect.) │
       └──────┬──────┘           │           └──────┬──────┘
              │                  │                  │
              ▼                  │                  ▼
       ┌─────────────┐           │           ┌─────────────┐
       │  distort     │           │           │  distort     │
       │ (lens dist₀) │          │           │ (lens dist₁) │
       └──────┬──────┘           │           └──────┬──────┘
              │                  │                  │
              ▼                  │                  ▼
       ┌─────────────┐           │           ┌─────────────┐
       │  K₀          │           │           │  K₁          │
       │ (intrinsics) │           │           │ (intrinsics) │
       └──────┬──────┘           │           └──────┬──────┘
              │                  │                  │
              ▼                  │                  ▼
         pixel (u₀,v₀)          │             pixel (u₁,v₁)
      in original image 0       │          in original image 1
              │                  │                  │
              │    RECTIFICATION │                  │
              │    ═════════════ │                  │
              ▼                  │                  ▼
       ┌─────────────┐           │           ┌─────────────┐
       │  K₀⁻¹        │          │           │  K₁⁻¹        │
       │ → normalised │           │           │ → normalised │
       └──────┬──────┘           │           └──────┬──────┘
              │                  │                  │
              ▼                  │                  ▼
       ┌─────────────┐           │           ┌─────────────┐
       │  undistort   │           │           │  undistort   │
       │ (invert      │           │           │ (invert      │
       │  dist₀)      │           │           │  dist₁)      │
       └──────┬──────┘           │           └──────┬──────┘
              │                  │                  │
              ▼                  │                  ▼
       ┌═════════════┐           │           ┌═════════════┐
       ║     R₀ᵀ     ║           │           ║     R₁ᵀ     ║
       ║ undo old    ║           │           ║ undo old    ║
       ║ orientation ║           │           ║ orientation ║
       ║             ║           │           ║             ║
       ║ PURE        ║           │           ║ PURE        ║
       ║ ROTATION    ║           │           ║ ROTATION    ║
       ╚══════╤══════╝           │           ╚══════╤══════╝
              │                  │                  │
              │         world direction             │
              │                  │                  │
              ▼                  │                  ▼
       ┌═════════════┐           │           ┌═════════════┐
       ║   R_rect    ║           │           ║   R_rect    ║
       ║ apply new   ║           │           ║ apply new   ║
       ║ orientation ║     SAME R_rect       ║ orientation ║
       ║             ║◄──── for both ────────║             ║
       ║ PURE        ║    cameras            ║ PURE        ║
       ║ ROTATION    ║           │           ║ ROTATION    ║
       ╚══════╤══════╝           │           ╚══════╤══════╝
              │                  │                  │
              ▼                  │                  ▼
       ┌─────────────┐           │           ┌─────────────┐
       │  ÷ z_r      │           │           │  ÷ z_r      │
       │ (reprojectn)│           │           │ (reprojectn)│
       └──────┬──────┘           │           └──────┬──────┘
              │                  │                  │
              ▼                  │                  ▼
       ┌─────────────┐           │           ┌─────────────┐
       │  K₀ (or P₀) │           │           │  K₁ (or P₁) │
       │ → pixel      │           │           │ → pixel      │
       └──────┬──────┘           │           └──────┬──────┘
              │                  │                  │
              ▼                  │                  ▼
       pixel (u₀',v₀')          │          pixel (u₁',v₁')
     in rectified image 0       │       in rectified image 1
              │                  │                  │
              │                  │                  │
              │          v₀' = v₁'                 │
              │    (SAME ROW guaranteed)            │
              │                  │                  │
              │    disparity d = u₀' - u₁'         │
              │    depth Z = f·b / d               │

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


$$ J = \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \begin{bmatrix} \nabla f_1^T \ \vdots \ \nabla f_m^T \end{bmatrix} = \begin{bmatrix}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_n} \
\vdots & \ddots & \vdots \
\frac{\partial y_m}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_n}
\end{bmatrix} $$

## 7. Map Starvation in Incremental SfM

### The Problem

In an incremental SfM pipeline, a fixed **maximum map size** cap can cause a
catastrophic failure mode called **map starvation**: the map fills up early in
the sequence, no new 3D points can be added, and as the camera moves forward
the existing points leave the field of view — leaving later frames with **zero
3D-2D correspondences**.

#### What happened (KITTI drive 0001, 114 frames)

The original stereo SfM had `MAX_MAP_POINTS = 10,000`.  Diagnostic
instrumentation revealed the following timeline:

| Frame range | Map size | 3D-2D matches per frame | Behaviour |
|---|---|---|---|
| 0–23 | 0 → 10,000 | 400–1,000+ | Normal growth |
| 24 | **10,000 (cap)** | 668 | Map full — no new points |
| 25–44 | 10,000 | 668 → 6 (decay) | Old points leaving FOV |
| **45–113** | 10,000 | **0** | **PnP fails, pose copied** |

The consequence: **61% of the sequence** (69 of 114 frames) had no 3D
localisation at all.  PnP fell back to copying the previous pose, so the
trajectory froze at ~53.6 m instead of the 107.3 m ground truth.

#### Why this is subtle

The failure is silent — no crash, no NaN.  The pipeline reports a plausible
trajectory that simply stops growing.  The map itself still has 10,000
perfectly good 3D points; they just happen to be behind the camera now.

A forward-moving car continuously discovers new scene geometry ahead while old
geometry disappears behind.  A static map of fixed size is fundamentally
incompatible with this.

### Formal Model

Let $N(t)$ be the number of map points visible at frame $t$, $N_{\text{new}}(t)$
the points newly triangulated, and $N_{\text{exit}}(t)$ the points that leave
the field of view.  At each frame:

$$N(t+1) = N(t) + N_{\text{new}}(t) - N_{\text{exit}}(t)$$

With a hard cap $N_{\max}$:

$$N_{\text{new}}(t) = 0 \quad \text{whenever } N(t) \ge N_{\max}$$

For a forward-driving sequence, $N_{\text{exit}}(t) > 0$ is roughly constant
(tied to the vehicle speed and FOV), so once the cap is hit:

$$N(t+k) = N_{\max} - k \cdot N_{\text{exit}} \xrightarrow{k \to \infty} 0$$

Map starvation is guaranteed for any bounded cap on a sufficiently long
forward-moving sequence.

### The Fix

Five changes were applied to `sfm_pipeline.py`:

#### 1. Raise the map cap (root cause)

```python
MAX_MAP_POINTS = 100_000   # was 10,000
```

Since the pipeline runs **pose-only BA** (3D points are fixed), computation
scales with the number of *observations*, not the number of points.  A 10×
larger map has negligible runtime impact when BA subsamples observations.

#### 2. Intermediate BA every 30 frames

```python
BA_INTERVAL = 30

# in add_frame():
if frame_id > 0 and frame_id % self.BA_INTERVAL == 0:
    self._run_intermediate_ba()
```

Without periodic BA, small per-frame pose errors compound.  By frame 60 the
drift may be large enough that 3D-2D matching fails even when points are
visible.  Intermediate BA corrects poses mid-sequence and keeps matching
healthy.

#### 3. Outlier culling

```python
def _cull_outlier_observations(self, max_reproj=8.0):
    # remove any observation with reprojection error > threshold
```

Even a few high-error observations ($> 10$ px) can bias BA.  Culling before
each BA run ensures the optimiser works on clean data.

#### 4. Two-round final BA (Huber → L2)

```
Round 1:  cull at 8 px  →  Huber-robust BA  (tolerates remaining outliers)
Round 2:  cull at 5 px  →  L2 BA            (precise refinement on clean obs)
```

The Huber loss in round 1 down-weights residuals larger than
$\delta = 1.0$ px:

$$\rho(r) = \begin{cases} \frac{1}{2} r^2 & |r| \le \delta \\ \delta(|r| - \frac{1}{2}\delta) & |r| > \delta \end{cases}$$

This makes the first pass robust to the remaining outliers that survived the
8 px cull.  After round 1, a tighter 5 px cull removes anything still off, and
the L2 pass gives the final precise fit.

#### 5. Point-lifetime tracking (`pt_last_seen`)

```python
self.pt_last_seen = {}  # pt_idx → last cam_idx that observed it
```

Every time a 3D point is matched in a new frame, its `pt_last_seen` entry is
updated.  This enables future map-maintenance strategies (e.g., evict points
not seen in 30+ frames) without requiring a full scan of the observation list.

### Results

| Strategy | End-to-End Distance | Error vs GT |
|---|---|---|
| Ground Truth | 107.34 m | — |
| No-Window SLAM | 114.19 m | +6.4% |
| Sliding-Window SLAM | 114.18 m | +6.4% |
| Stereo SfM (before fix) | **53.56 m** | **−50.1%** |
| Stereo SfM (after fix) | **113.34 m** | **+5.6%** |

The fixed SfM now produces the trajectory closest to ground truth among all
three strategies, which is expected: global BA over all 114 frames should
outperform local/incremental optimisation, provided it actually *sees* all
114 frames — which the map starvation bug prevented.

### Takeaway

A map cap is a reasonable safeguard, but it must be sized relative to the
**expected sequence length and scene coverage**.  For a 114-frame forward drive
at 10 Hz (11.4 seconds, ~107 m), new ORB features enter the FOV at ~400–600
per frame.  Even with substantial overlap, the map needs to grow well beyond
10K to maintain continuous localisation.  The safe rule of thumb:

$$N_{\max} \gg T \cdot N_{\text{new/frame}}$$

where $T$ is the total number of frames.  For this sequence: $N_{\max} \gg 114 \times 500 \approx 57{,}000$, consistent with the 43,805 points the fixed
pipeline actually accumulated.

## 8. Adaptive Intermediate BA — Eliminating Dataset-Specific Hyperparameters

### The Problem with Fixed-Interval BA

The initial map-starvation fix (§7) introduced `BA_INTERVAL = 30` — run
intermediate Bundle Adjustment every 30 frames.  While effective on KITTI drive
0001 (114 frames, 10 Hz, ~1 m/s), this constant is dataset-specific:

| Scenario | Ideal interval | Why 30 fails |
|---|---|---|
| Highway @ 30 m/s | ~5–10 frames | Drift accumulates 3× faster; BA at 30 is too late |
| Static indoor scan | ~100+ frames | BA every 30 wastes compute on negligible drift |
| Sharp turns | Varies per turn | Fixed schedule misses the exact frames that need it |

A general-purpose SfM pipeline cannot assume it knows the motion dynamics
ahead of time.

### How Real SfM Systems Decide When to Run BA

The key insight: **the pipeline already computes all the information it needs
to detect drift — it just wasn't using it.**

After each frame's PnP solve, OpenCV returns the inlier set.  From the inlier
3D-2D correspondences and the solved pose, the **mean reprojection error** is:

$$\bar{e}_t = \frac{1}{|\mathcal{I}_t|} \sum_{i \in \mathcal{I}_t} \|\pi(K, R_t, t_t, \mathbf{X}_i) - \mathbf{u}_i\|_2$$

where $\pi$ is the projection function, $\mathcal{I}_t$ is the PnP inlier
set, $\mathbf{X}_i$ are the 3D map points, and $\mathbf{u}_i$ are the
observed 2D keypoints.

This error rises when:
- Accumulated pose drift causes the map and current pose to disagree
- Outlier 3D points bias the PnP solution
- Scene geometry changes rapidly (sharp turns, elevation changes)

All of these are exactly the situations where BA is needed.

### The Adaptive Trigger

Replace the fixed interval with three parameters:

```python
BA_REPROJ_THRESH = 2.0   # trigger BA when mean PnP reproj error (px) exceeds this
BA_MIN_INTERVAL  = 10    # never run BA more often than every 10 frames
BA_MAX_INTERVAL  = 50    # always run BA at least every 50 frames (safety net)
```

The decision rule after each frame:

```python
frames_since_ba = frame_id - last_ba_frame
trigger_ba = (
    (frames_since_ba >= BA_MIN_INTERVAL and mean_reproj_err > BA_REPROJ_THRESH)
    or frames_since_ba >= BA_MAX_INTERVAL
)
```

This adapts to any dataset:

| Condition | Behaviour |
|---|---|
| $\bar{e}_t > 2.0$ px and $\ge 10$ frames since last BA | **Error trigger** — drift detected, correct now |
| $\bar{e}_t \le 2.0$ px for 50 consecutive frames | **Safety net** — correct anyway as precaution |
| $\bar{e}_t > 2.0$ px but $< 10$ frames since last BA | **Cooldown** — avoid thrashing the optimiser |

### Why These Thresholds Are General

- **2.0 px** reprojection threshold: ORB keypoint localisation has ~0.5–1.0 px
  noise.  A mean error of 2.0 px is 2–4× the noise floor — a clear signal that
  systematic drift (not just noise) is present.  This holds for any
  pinhole-model camera regardless of resolution or focal length, because the
  PnP inlier threshold (4.0 px) already filters gross outliers.

- **10-frame minimum**: BA on $N$ cameras with $M$ subsampled observations
  takes $O(M)$ time (pose-only, points fixed).  Running it every frame would
  double total compute for marginal benefit.  10 frames is a small-enough
  minimum that even fast-accumulating drift is caught within $\sim$1 second at
  10 Hz.

- **50-frame maximum**: Even when reprojection error stays low, small
  systematic biases can accumulate invisibly.  50 frames (5 seconds at 10 Hz)
  bounds the worst-case uncorrected drift without imposing significant overhead.

### Results on KITTI Drive 0001

| | Fixed (`BA_INTERVAL=30`) | Adaptive |
|---|---|---|
| BA at frames | 30, 60, 90 | 50, 100 |
| Number of BA runs | 3 | 2 |
| Trigger type | Fixed clock | Max-interval safety net |
| Final trajectory | 113.34 m | 113.34 m |

On this easy sequence (slow, roughly straight), reprojection error stays below
2.0 px throughout — drift is minimal.  The adaptive trigger correctly defers
BA to frames 50 and 100 (the safety net), saving one BA run while producing an
identical result.

On a harder sequence (e.g., sharp turns at higher speed), the error threshold
would fire earlier and more frequently, providing correction exactly when the

## 9. PGO Overcorrection & Back-End Comparison (Drive 0005)

All three pipelines share the **same stereo front-end**: ORB feature detection,
stereo matching, disparity-based triangulation, and PnP pose estimation. They
differ **only** in the back-end optimisation applied after the front-end
produces an initial pose.

Running them on a **different drive** (2011_09_26 drive 0005, 160 frames)
reveals a critical failure mode of the sliding-window back-end.

### 9.1 Results on Drive 0005 (Before Fix)

| Method | End-to-End Distance | Error vs GT |
|---|---|---|
| Ground Truth | 67.62 m | — |
| No-Window Stereo SLAM | 70.76 m | +4.6% |
| **Sliding-Window SLAM** | **41.74 m** | **−38.3%** |
| Stereo SfM (global BA) | 67.48 m | −0.2% |

The sliding-window SLAM **collapsed** to 62% of the true path length, while
no-window SLAM and SfM both stayed within 5%.

### 9.2 Back-End Comparison

All three share the same stereo front-end (ORB + stereo triangulation + PnP).
They differ in **what the back-end optimises** and **when**:

```
┌──────────────────────────────────────────────────────────────────────┐
│                   No-Window Stereo SLAM                              │
│                                                                      │
│  Frame 0 ──PnP──▶ Frame 1 ──PnP──▶ Frame 2 ──PnP──▶ … ──▶ Frame N │
│                                                                      │
│  • Each frame: stereo-triangulate 3D pts → PnP to next frame        │
│  • Once a pose is computed, it is NEVER revisited                    │
│  • Drift accumulates but is bounded by stereo metric scale           │
│  • No back-end optimisation at all                                   │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                 Sliding-Window Stereo SLAM                            │
│                                                                      │
│  Tier 1 — Local BA:                                                  │
│    Optimise the last W keyframe poses (pose-only, points fixed)      │
│    Runs at every new keyframe insertion                               │
│                                                                      │
│  Tier 2 — Loop Closure + PGO:                                        │
│    1. Match new KF descriptors vs all old KFs (BF Hamming, d<50)     │
│    2. Geometric verify with RANSAC PnP (≥15 inliers)                │
│    3. Add loop edge → run full pose-graph optimisation                │
│                                                                      │
│  KF0 ──odom──▶ KF1 ──odom──▶ KF2 ──odom──▶ … ──odom──▶ KFN         │
│   ╰─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ loop edge ─ ─ ─ ─ ─ ─ ─ ─ ─╯         │
│                                                                      │
│  PGO distributes the loop constraint across ALL keyframe poses       │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                      Stereo SfM                                      │
│                                                                      │
│  Forward pass: frame-by-frame stereo triangulation + PnP             │
│  (same front-end as no-window SLAM, plus adaptive intermediate BA)   │
│                                                                      │
│  Final: Global Bundle Adjustment over ALL cameras + ALL 3D points    │
│    • Round 1: Huber loss (robust to outliers)                        │
│    • Outlier cull at 5.0 px                                          │
│    • Round 2: L2 loss (fine-tune)                                    │
│                                                                      │
│  Key difference: optimises 3D structure jointly with poses           │
│  → self-correcting; no need for explicit loop closure                │
└──────────────────────────────────────────────────────────────────────┘
```

### 9.3 PGO Overcorrection — The Perceptual Aliasing Problem

**Root cause**: on a one-way drive the vehicle never revisits the same place,
yet the loop closure detector fires 4 times:

| Loop Edge | KFs | Inliers | Approx. Frames |
|---|---|---|---|
| 1 | KF 5 ↔ KF 15 | 21 | ~15 ↔ ~45 |
| 2 | KF 5 ↔ KF 16 | 20 | ~15 ↔ ~48 |
| 3 | KF 34 ↔ KF 49 | 18 | ~102 ↔ ~147 |
| 4 | KF 33 ↔ KF 50 | 15 | ~99 ↔ ~150 |

These are **perceptual aliases** — distinct locations that look similar because
urban roads contain repetitive structure (lane markings, poles, curbs, building
facades). The ORB descriptor match (Hamming distance < 50) and even 15–21 PnP
RANSAC inliers are not enough to distinguish a true revisit from a look-alike.

**What PGO does with a false loop edge**:

The pose-graph cost for an edge $(i, j)$ with measured relative transform
$T_{ij}^{\text{meas}}$ is:

$$
e_{ij} = \log\!\bigl( (T_{ij}^{\text{meas}})^{-1} \cdot T_i^{-1} \cdot T_j \bigr)
$$

PGO minimises $\sum_{\text{edges}} \|e_{ij}\|^2$. A false loop between KF 5
and KF 15 says "these two poses should be related by $T_{ij}^{\text{meas}}$"
— but they are actually 30 m apart. PGO **distributes this error** across all
intermediate keyframes, pulling the entire trajectory inward:

```
True trajectory (one-way drive):
  ●───●───●───●───●───●───●───●───●───●───●───●──▶  (67.62 m)

After false loop (KF5↔KF15): PGO contracts the chain
  ●───●──●─●●●●●──●───●───●───●───●──▶              (41.74 m)
           ↑
       KF 5–15 pulled together
```

The contraction is severe because:

1. **No reciprocal constraint**: on a one-way drive, there is no opposing loop
   edge to push the poses back apart.
2. **Multiple false loops compound**: loops 1–2 contract the first third; loops
   3–4 contract the last third.  The trajectory shrinks from both ends.
3. **Odometry edges are soft**: PGO treats odometry and loop edges with equal
   weight.  A single false loop can overpower dozens of correct odometry edges.

### 9.4 Why the Other Two Stereo Pipelines Are Immune

**No-window stereo SLAM** — same stereo front-end, but **no back-end** at all.
Each pose is computed from the immediately preceding frame's stereo-triangulated
3D points via PnP, then never revisited. Errors only propagate forward as small
frame-to-frame drift (~0.05 m per frame), and the stereo baseline gives metric
scale — so drift stays bounded. On drive 0005: +4.6% overshoot.

**Stereo SfM** — same stereo front-end, but the back-end is **global Bundle
Adjustment** over all cameras and all 3D points jointly. If two distant camera
poses share 3D points (the analogue of a "loop"), BA automatically exploits
this — but it does so by adjusting all connected points and cameras
simultaneously, which naturally rejects false associations (they produce large
residuals that get down-weighted by Huber loss and then culled). Result: −0.2%
error.

### 9.5 Implemented Fixes

All three mitigations from the table above were implemented:

1. **Odometry consistency check**: Before accepting a loop edge, compute the
   odometry distance between the two keyframes and the loop-derived
   translation.  If `loop_t < 0.5 × odom_dist` when `odom_dist > 5 m`,
   reject as perceptual alias.

2. **Raised PnP inlier threshold**: 15 → 25.

3. **Robust PGO**: Huber loss (`f_scale=0.1`) with per-edge weights
   (odometry=1.0, loop=0.3).

### 9.6 Results After Fix

**Drive 0005** (160 frames) — all 4 false loops rejected:

| Method | Before Fix | After Fix | GT |
|---|---|---|---|
| No-Window | 70.76 m | 70.76 m | 67.62 m |
| **Sliding-Window** | **41.74 m (−38%)** | **70.76 m (+4.6%)** | 67.62 m |
| SfM | 67.48 m | 67.48 m | 67.62 m |

**Drive 0018** (276 frames, city sequence with actual revisits):

| Method | End-to-End | Error vs GT |
|---|---|---|
| Ground Truth | 46.53 m | — |
| No-Window SLAM | 29.94 m | **−35.6%** |
| **Sliding-Window SLAM** | **43.67 m** | **−6.1%** |
| Stereo SfM | 45.10 m | −3.1% |
| Sliding-window stats | 49 keyframes, 4 true loop closures |

On drive 0018, the vehicle passes through similar areas, producing **genuine
loop closures** (KF 12↔25, KF 3↔26, KF 0↔27, KF 0↔28).  The odometry
consistency check rejected ~130 perceptual aliases and accepted these 4.
PGO then distributed the corrections, recovering the trajectory from 29.94 m
(no-window, −35.6%) to 43.67 m (sliding-window, −6.1%).

This demonstrates the core value proposition: **with more data and real
revisits, the sliding-window back-end significantly outperforms no-window.**

### 9.7 Why Cross-Keyframe Local BA Doesn't Help (Pose-Only)

Attempts to add cross-keyframe constraints to local BA — matching keyframe
K's features against keyframe K−1's 3-D points — actually *degraded*
results (trajectory contracted to 27–46 m).  Root cause:

- `stereo_pose_refine` is **pose-only** (3-D points are held fixed)
- Keyframe K−1's 3-D points carry its accumulated pose drift
- Optimising K's pose against K−1's drifted points **imports** that bias

For cross-keyframe constraints to be effective, one needs **joint
structure+pose BA** (optimise both 3-D points and camera poses
simultaneously), which is precisely what global SfM BA does.  The sliding-
window back-end's strength therefore comes entirely from loop closure +
PGO, not from pose-only local BA.

### 9.8 Summary — Same Stereo Front-End, Different Back-Ends

All three use: **ORB features → stereo matching → disparity triangulation → PnP**.

| Property | No-Window | Sliding-Window | SfM (Global BA) |
|---|---|---|---|
| Front-end | Stereo ORB + PnP | Stereo ORB + PnP | Stereo ORB + PnP |
| **Back-end** | **None** | **Local BA + PGO** | **Global BA (cams + pts)** |
| Loop closure | ✗ | ✓ (with odometry check) | Implicit (shared 3D points) |
| Perceptual alias defence | Immune | Odom. consistency + Huber PGO | Huber + outlier cull |
| |
| Drive 0001 (straight, 114 fr) | 114.19 m (+6.4%) | 114.18 m (+6.4%) | 113.34 m (+5.6%) |
| Drive 0005 (turns, 160 fr) | 70.76 m (+4.6%) | 70.76 m (+4.6%) | **67.48 m (−0.2%)** |
| **Drive 0018 (city, 276 fr)** | **29.94 m (−35.6%)** | **43.67 m (−6.1%)** | **45.10 m (−3.1%)** |
| Drive 0009 (residential, 453 fr) | 266.31 m (+6.5%) | **258.59 m (+3.4%)** | 219.17 m (−12.3%) |
| Drive 0051 (highway, 444 fr) | 249.73 m (−1.6%) | **248.31 m (−2.1%)** | 215.03 m (−15.2%) |
| |
| Best suited for | Short sequences, real-time | Longer sequences with revisits | Offline reconstruction |

#### Observations across 5 drives

1. **Sliding-window SLAM improves with sequence length.**  On short drives
   (0001, 0005) it matches no-window exactly — too few keyframes for
   meaningful loop closures.  On longer drives (0018, 0009, 0051) it
   consistently outperforms no-window, with 23 and 17 true loop closures
   accepted on drives 0009 and 0051 respectively.

2. **SfM global BA undershoots on long sequences.**  On the two longest
   drives (0009 and 0051, ~450 frames each) SfM trajectory length is
   12–15% shorter than ground truth.  Global BA over-constrains the
   structure when the point cloud grows large, pulling cameras inward.
   On shorter sequences (≤276 frames) it remains competitive.

3. **No-window odometry is surprisingly robust** on highway/residential
   drives with smooth motion (0009, 0051) but collapses on city driving
   with sharp turns and stops (0018: −35.6%).

4. **The odometry consistency check is essential.**  Without it, drives
   0009 and 0051 would accept hundreds of false loop closures from
   perceptual aliasing (similar road textures), causing PGO to
   overcorrect — the exact failure mode documented in §9.3 for drive 0005.