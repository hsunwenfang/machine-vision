

# Programming Tech Spec

## 1. Core Libraries
### 1.1 OpenCV (cv2) - Primary Geometric Backend
*   **Role**: Image processing, calibration, feature extraction (ORB/SIFT), geometric solvers (PnP, RANSAC).
*   **Performance**: Python wrappers call optimized C++. No custom C++ extensions related to standard CV algorithms are expected unless custom factor graph optimization is required.

### 1.2 PyTorch / TensorFlow - Deep Learning Backend
*   **Role**: Advanced feature extraction (SuperPoint) or Dense Depth (RAFT-Stereo).
*   **Recommendation**: **PyTorch** is preferred for modern CV research implementations.
*   **Hardware Acceleration**:
    *   **x86 (NVIDIA)**: CUDA support is mature.
    *   **Mac (M1/M2/M3)**: PyTorch supports `mps` (Metal Performance Shaders) for GPU acceleration.

---

# Computer Vision Tech Spec

## 1. Camera Calibration
### 1.1 Objectives
*   Recover Intrinsic Matrix ($K$) and Distortion Coefficients ($D$).
*   Recover Stereo Baseline ($b$) and Rectification Transforms ($R_{rect}, P_{rect}$).

### 1.2 Pipeline
1.  **Detection**: Robust Checkerboard Detection (Iterative `findChessboardCornersSB` with robust standard fallback).
2.  **Optimization**:
    *   Constrained Solver: Fix Principal Point, Fix Aspect Ratio, Zero Tangent Distortion (for small ROIs).
    *   Refinement: Minimize reprojection error over all frames.

## 2. Stereo SLAM / Visual Odometry

### 2.1 Pipeline Architecture
**Sequence**: `Feature Extraction` -> `Matching` -> `Pose Estimation (Sticking)` -> `Local Mapping`

### 2.2 Modules

#### A. Feature Extraction
*   **Option 1 (Fast/Standard)**: classical ORB (Oriented FAST and Rotated BRIEF).
*   **Option 2 (Robust)**: SIFT (Scale-Invariant Feature Transform).
*   **Option 3 (Deep Learning)**: SuperPoint (requires GPU).

#### B. Correspondence / Matching
*   **Temporal (Tracking)**: Matching Frame $N-1$ to $N$.
    *   Method: `BFMatcher` (Hamming for binary, L2 for float) + Cross-Check.
*   **Stereo (Depth)**: Matching Left $N$ to Right $N$.
    *   Constraint: Epipolar Line ($y_L \approx y_R$) + Disparity Order ($x_L > x_R$).

#### C. Pose Estimation ("Stitching")
1.  **Initialization**: 
    *   Perspective-n-Point (PnP) problem.
    *   Input: 3D World Points (from $N-1$) + 2D Image Points (from $N$).
2.  **Robust Estimation**:
    *   **RANSAC**: Outlier rejection loop.
    *   **Refinement**: 
        *   Standard: Levenberg-Marquardt (Minimizes L2 norm).
        *   **Advanced**: **IRLS** (Iteratively Reweighted Least Squares) with Huber/Tukey loss to suppress outliers.

#### D. Depth Calculation
*   **Sparse**: Triangulation of inlier feature points ($Z = f \cdot b / d$).
*   **Dense (Future Work)**: Semi-Global Matching (SGBM) for full depth maps.

#### E. Optimization (Backend)
*   **Local Bundle Adjustment**: Refine last $K$ poses and 3D points simultaneously.
*   **Loop Closure**: Detect previously visited locations (Bag of Words) to correct drift.

##### E.1 Stereo Bundle Adjustment Modes

The stereo rig provides two views per timestep. The Bundle Adjustment step can leverage one or both views. Three optimization modes are supported:

| Mode | Optimized Variables | Derived Variables | Description |
|------|---|---|---|
| **Left-Only** | $T_L$, $\mathbf{P}_{3D}$ | $T_R = T_{stereo} \cdot T_L$ | Minimize reprojection error in the **left image only**. The right camera pose is computed from the fixed stereo extrinsic $T_{stereo}$. This is the standard VO approach. |
| **Right-Only** | $T_R$, $\mathbf{P}_{3D}$ | $T_L = T_{stereo}^{-1} \cdot T_R$ | Minimize reprojection error in the **right image only**. The left camera pose is derived via the inverse stereo extrinsic. Useful for validation or when the left image is degraded. |
| **Joint (Recommended)** | $T_L$, $\delta T_{stereo}$, $\mathbf{P}_{3D}$ | $T_R = (T_{stereo} + \delta T_{stereo}) \cdot T_L$ | Minimize reprojection error in **both images simultaneously**. The stereo extrinsic is treated as a refinable parameter (initialized from calibration, with a small perturbation $\delta T_{stereo}$). This enforces the **rigid body constraint** while allowing the optimizer to compensate for calibration drift or mechanical vibration. |

**Notation**:
*   $T_L$: Left camera pose (world-to-camera), 6 DoF (rotation + translation).
*   $T_R$: Right camera pose, 6 DoF.
*   $T_{stereo}$: Extrinsic transformation from Left to Right camera (from calibration).
*   $\delta T_{stereo}$: Small refinement to the stereo extrinsic, 6 DoF.
*   $\mathbf{P}_{3D}$: Set of 3D landmark points (structure), 3 DoF per point.
*   $K$: Intrinsic matrix — **fixed** (not optimized; obtained from calibration).

**Cost Function (Joint Mode)**:

$$\min_{T_L, \delta T_S, \mathbf{P}} \sum_{i} \left\| \mathbf{u}^L_i - \pi(K, T_L, \mathbf{P}_i) \right\|^2 + \sum_{i} \left\| \mathbf{u}^R_i - \pi(K, (T_S + \delta T_S) \cdot T_L, \mathbf{P}_i) \right\|^2$$

where $\pi(K, T, \mathbf{P})$ is the pinhole projection function and $\mathbf{u}^L_i, \mathbf{u}^R_i$ are the observed 2D feature locations in the left and right images respectively.

**Implementation**: See `src/bundle_adjustment.py` — `reprojection_error_stereo()`.

**Hardware**: [TODO] Can GPU speedup computing Jacobian in BA and avoid memory issue

    4-1. VO, Optical Flow, 