import taichi as ti

@ti.func
def quaternion_to_rotation_ti(q: ti.math.vec4) -> ti.math.mat3:
    """Convert a quaternion to a rotation matrix.

    Args:
        q (ti.math.vec4): Quaternion represented as (w, x, y, z).

    Returns:
        ti.math.mat3: Corresponding rotation matrix.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)
    
    return ti.math.mat3([[r00, r01, r02],
                         [r10, r11, r12],
                         [r20, r21, r22]])
    

@ti.func
def quaternion_translation_to_transform_ti(
    q: ti.math.vec4,
    t: ti.math.vec3
) -> ti.math.mat4:
    """Convert a quaternion and translation vector to a 4x4 transformation matrix.

    Args:
        q (ti.math.vec4): Quaternion represented as (w, x, y, z).
        t (ti.math.vec3): Translation vector.

    Returns:
        ti.math.mat4: Corresponding 4x4 transformation matrix.
    """
    R = quaternion_to_rotation_ti(q)
    
    transform = ti.math.mat4([[R[0, 0], R[0, 1], R[0, 2], t[0]],
                               [R[1, 0], R[1, 1], R[1, 2], t[1]],
                               [R[2, 0], R[2, 1], R[2, 2], t[2]],
                               [0.0,      0.0,      0.0,     1.0]])
    return transform


@ti.kernel
def quaternion_translation_to_transform_batch_ti(
    quaternions: ti.types.ndarray(dtype=ti.f32, ndim=2), # (N, 4)
    translations: ti.types.ndarray(dtype=ti.f32, ndim=2), # (N, 3)
    transforms: ti.types.ndarray(dtype=ti.f32, ndim=3) # (N, 4, 4)
):
    N = quaternions.shape[0]
    
    for i in range(N):
        q = ti.math.vec4([quaternions[i, 0], quaternions[i, 1], quaternions[i, 2], quaternions[i, 3]])
        t = ti.math.vec3([translations[i, 0], translations[i, 1], translations[i, 2]])
        
        transform = quaternion_translation_to_transform_ti(q, t)
        
        for r, c in ti.static(ti.ndrange(4, 4)):
            transforms[i, r, c] = transform[r, c]


@ti.func
def quaternion_rotation_ti(
    quaternion_source: ti.math.vec4,
    quaternion_source_to_target: ti.math.vec4
) -> ti.math.vec4:
    """Compute the rotation quaternion from source to target frame.

    Args:
        quaternion_source (ti.math.vec4): Quaternion of the source frame.
        quaternion_source_to_target (ti.math.vec4): Quaternion from source to target frame.

    Returns:
        ti.math.vec4: Rotation quaternion from source to target frame.
    """
    w1, x1, y1, z1 = quaternion_source[0], quaternion_source[1], quaternion_source[2], quaternion_source[3]
    w2, x2, y2, z2 = quaternion_source_to_target[0], quaternion_source_to_target[1], quaternion_source_to_target[2], quaternion_source_to_target[3]
    
    w = w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1
    x = w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1
    y = w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1
    z = w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1
    
    return ti.math.vec4([w, x, y, z])


@ti.func
def quaternion_transform_point_ti(
    point: ti.math.vec3,
    quaternion: ti.math.vec4
) -> ti.math.vec3:
    """Rotate a point using a quaternion.

    Args:
        point (ti.math.vec3): Point to be rotated.
        quaternion (ti.math.vec4): Quaternion representing the rotation.

    Returns:
        ti.math.vec3: Rotated point.
    """
    q_conj = ti.math.vec4([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])
    
    p_as_quat = ti.math.vec4([0.0, point[0], point[1], point[2]])
    
    q_p = quaternion_rotation_ti(quaternion_source=p_as_quat, quaternion_source_to_target=quaternion)
    rotated_p_as_quat = quaternion_rotation_ti(quaternion_source=q_p, quaternion_source_to_target=q_conj)
    
    return ti.math.vec3([rotated_p_as_quat[1], rotated_p_as_quat[2], rotated_p_as_quat[3]])


@ti.func
def transform_reverse_ti(transform: ti.math.mat4) -> ti.math.mat4:
    """Compute the inverse of a 4x4 transformation matrix.

    Args:
        transform (ti.math.mat4): Input 4x4 transformation matrix.

    Returns:
        ti.math.mat4: Inverse of the input transformation matrix.
    """
    R = ti.math.mat3([[transform[0, 0], transform[0, 1], transform[0, 2]],
                      [transform[1, 0], transform[1, 1], transform[1, 2]],
                      [transform[2, 0], transform[2, 1], transform[2, 2]]])
    
    t = ti.math.vec3([transform[0, 3], transform[1, 3], transform[2, 3]])
    
    R_inv = R.transpose()
    t_inv = -R_inv @ t

    transform_inv = ti.math.mat4(
        [[R_inv[0, 0], R_inv[0, 1], R_inv[0, 2], t_inv[0]],
        [R_inv[1, 0], R_inv[1, 1], R_inv[1, 2], t_inv[1]],
        [R_inv[2, 0], R_inv[2, 1], R_inv[2, 2], t_inv[2]],
        [0.0,          0.0,          0.0,        1.0]]
    )
    
    return transform_inv

@ti.func
def transform_torch_to_ti(transform: ti.types.ndarray(dtype=ti.f32, ndim=2)) -> ti.math.mat4:   
    """Convert a PyTorch tensor transformation matrix to a Taichi matrix.

    Args:
        transform (torch.Tensor): Input 4x4 transformation matrix as a PyTorch tensor.

    Returns:
        ti.math.mat4: Corresponding 4x4 transformation matrix in Taichi.
    """
    mat = ti.math.mat4()
    for idx in ti.static(range(16)):
        i = idx // 4
        j = idx % 4
        mat[i, j] = float(transform[i, j].item())
    return mat


@ti.func
def project_point_to_camera(
    point_homogeneous: ti.math.vec4,
    transformer_world_camera: ti.math.mat4,
    camera_intrinsic: ti.math.mat3,
)-> ti.math.vec3:
    """_summary_

    Args:
        point_homogeneous (ti.math.vec4): point in world homogeneous coordinates
        transformer_world_camera (ti.math.mat4): Transformation matrix from world to camera
        camera_intrinsic (ti.math.mat3): _camera intrinsic matrix

    Returns:
        ti.math.vec3: uv in image plane and depth in camera coordinates
    """
    point_camera = transformer_world_camera @ point_homogeneous

    uv_camera = camera_intrinsic @ ti.math.vec3([
        point_camera[0] / (point_camera[2] + 1e-8),
        point_camera[1] / (point_camera[2] + 1e-8),
        1.0
    ])
    
    return ti.math.vec3([
        uv_camera[0], uv_camera[1], point_camera[2]
    ])


@ti.func
def compute_gaussian_covariance(
    scale: ti.math.vec3,
    rotation: ti.math.vec4
)-> ti.math.mat3:
    """Compute the covariance matrix of a Gaussian sphere given its scale and rotation.

    Args:
        scale (ti.math.vec3): Scale of the Gaussian sphere along x, y, z axes.
        rotation (ti.math.vec4): Rotation of the Gaussian sphere represented as a quaternion.

    Returns:
        ti.math.mat3: Covariance matrix of the Gaussian sphere.
    """
    R = quaternion_to_rotation_ti(rotation)
    S = ti.math.mat3([[scale[0] * scale[0], 0.0, 0.0],
                      [0.0, scale[1] * scale[1], 0.0],
                      [0.0, 0.0, scale[2] * scale[2]]])
    
    covariance = R @ S @ S.transpose() @ R.transpose()
    
    return covariance


@ti.func
def det_3x3_covariance_ti(S):
    a, b, c = S[0,0], S[0,1], S[0,2]
    d, e = S[1,1], S[1,2]
    f = S[2,2]
    
    return (a*d*f + 2*b*c*e - a*e*e - d*c*c - f*b*b)


@ti.func
def gaussian_density_ti(
    diff: ti.math.vec3,
    covariance: ti.math.mat3,
) -> ti.f32:
    """Compute the density of a Gaussian sphere at a given position.

    Args:
        pos (ti.math.vec3): Position where the density is evaluated.
        mean (ti.math.vec3): Mean of the Gaussian sphere.
        covariance (ti.math.mat3): Covariance matrix of the Gaussian sphere.

    Returns:
        ti.f32: Density of the Gaussian sphere at the given position.
    """
    
    exponent = -0.5 * (diff.transpose() @ covariance.inverse() @ diff)
    det_cov = det_3x3_covariance_ti(covariance)
    normalization = 1.0 / ti.sqrt((2 * 3.14159265) ** 3 * covariance.determinant())
    
    return normalization * ti.exp(exponent)


@ti.func
def gaussian_density_cov_inv_ti(
    diff: ti.math.vec3,
    cov_inv: ti.math.mat3,
    cov_det: ti.f32
) -> ti.f32:
    """Compute the density of a Gaussian sphere at a given position using pre-computed inverse covariance and determinant.

    Args:
        pos (ti.math.vec3): Position where the density is evaluated.
        mean (ti.math.vec3): Mean of the Gaussian sphere.
        cov_inv (ti.math.mat3): Inverse of the covariance matrix of the Gaussian sphere.
        cov_det (ti.f32): Determinant of the covariance matrix of the Gaussian sphere.

    Returns:
        ti.f32: Density of the Gaussian sphere at the given position.
    """
    
    exponent = -0.5 * (diff.transpose() @ cov_inv @ diff)
    normalization = 1.0 / ti.sqrt((2 * 3.14159265) ** 3 * cov_det)
    
    return normalization * ti.exp(exponent)


@ti.func
def unnormalized_gaussian_density_ti(
    diff: ti.math.vec3,
    covariance: ti.math.mat3,
) -> ti.f32:
    """Compute the density of a Gaussian sphere at a given position.

    Args:
        pos (ti.math.vec3): Position where the density is evaluated.
        mean (ti.math.vec3): Mean of the Gaussian sphere.
        covariance (ti.math.mat3): Covariance matrix of the Gaussian sphere.

    Returns:
        ti.f32: Density of the Gaussian sphere at the given position.
    """    
    exponent = -0.5 * (diff.transpose() @ covariance.inverse() @ diff)
    
    return ti.exp(exponent)

@ti.func
def compute_gaussian_normalized_factor_cov_inv_ti(
    cov_inv: ti.math.mat3
)-> ti.f32:
    det_cov = 1.0 / cov_inv.determinant()
    confidence = 0.0
    
    if det_cov > 1e-8:
        confidence = 1.0 / ti.sqrt((2 * 3.14159265) ** 3 * det_cov)
    
    return confidence
