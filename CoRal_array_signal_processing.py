import numpy as np
from numpy.linalg import eig, pinv
import matplotlib.pyplot as plt

# ==========================================================
# ESPRIT FUNCTION
# ==========================================================
def esprit(X, fc, d, c, num_sources=1):
    """
    ESPRIT angle estimation
    X : M x N data matrix
    """
    M, N = X.shape

    R = X @ X.conj().T / N
    eigvals, eigvecs = eig(R)

    idx = np.argsort(np.abs(eigvals))[::-1]
    Es = eigvecs[:, idx[:num_sources]]

    Es1 = Es[:-1, :]
    Es2 = Es[1:, :]

    Psi = pinv(Es1) @ Es2
    phi = eig(Psi)[0]

    angles = []
    for p in phi:
        phase = np.angle(p)
        theta = np.arcsin(-phase * c / (2*np.pi*fc*d))
        angles.append(np.rad2deg(theta.real))

    return np.array(angles)


# ==========================================================
# MUSIC FUNCTION
# ==========================================================
def music(X, fc, d, c, num_sources=1, scan_angles=np.linspace(-90,90,1000)):
    """
    MUSIC angle estimation
    """
    M, N = X.shape
    R = X @ X.conj().T / N

    eigvals, eigvecs = eig(R)
    idx = np.argsort(np.abs(eigvals))[::-1]

    En = eigvecs[:, idx[num_sources:]]

    P_music = []

    for theta_deg in scan_angles:
        theta = np.deg2rad(theta_deg)
        phi = np.exp(-1j*2*np.pi*fc*d*np.sin(theta)/c)
        a = np.array([phi**k for k in range(M)]).reshape(-1,1)

        spectrum = 1 / np.abs(a.conj().T @ En @ En.conj().T @ a)
        P_music.append(spectrum[0,0].real)

    P_music = np.array(P_music)
    est_angle = scan_angles[np.argmax(P_music)]

    return est_angle, scan_angles, P_music


# ==========================================================
# CoRaL RAW MODEL GENERATION
# ==========================================================
def generate_coral_data(theta_deg, SNR_dB=10, N=100):

    c = 3e8
    f0 = 2.4e9
    f5 = 1e6
    fc = f0 + f5
    d = 0.05
    M = 4

    theta = np.deg2rad(theta_deg)

    # CFO
    f = np.array([0, 1e3, 2e3, 3e3, 4e3])
    Delta_t = 1e-6

    d_raw = []

    for k in range(1, M+1):

        geom = np.exp(
            -1j*2*np.pi*fc*(k-1)*d*np.sin(theta)/c
        )

        cfo = np.exp(
            -1j*2*np.pi*(f[1]-f[k])*Delta_t
        )

        Mk = np.exp(-1j*2*np.pi*0.1)

        d_raw.append(geom * cfo * Mk)

    d_raw = np.array(d_raw).reshape(-1,1)

    # Normalize (remove CFO & Mk)
    d_norm = []

    for k in range(1, M+1):

        cfo_comp = np.exp(+1j*2*np.pi*(f[1]-f[k])*Delta_t)
        Mk = np.exp(-1j*2*np.pi*0.1)

        d_norm.append(d_raw[k-1] * cfo_comp / Mk)

    d_norm = np.array(d_norm).reshape(-1,1)

    # Snapshots
    s = (np.random.randn(1,N)+1j*np.random.randn(1,N))/np.sqrt(2)

    sigma2 = 10**(-SNR_dB/10)
    noise = np.sqrt(sigma2/2)*(
        np.random.randn(M,N)+1j*np.random.randn(M,N)
    )

    X = d_norm @ s + noise

    return X, fc, d, c


# ==========================================================
# VALIDATION
# ==========================================================
if __name__ == "__main__":
    theta_true = 30

    X, fc, d, c = generate_coral_data(theta_true, SNR_dB=15)

    # ESPRIT
    theta_esprit = esprit(X, fc, d, c)[0]

    # MUSIC
    theta_music, scan_angles, P = music(X, fc, d, c)

    print("True angle :", theta_true)
    print("ESPRIT     :", theta_esprit)
    print("MUSIC      :", theta_music)

    # Plot MUSIC spectrum
    plt.plot(scan_angles, 10 * np.log10(P / np.max(P)))
    plt.title("MUSIC Spectrum")
    plt.xlabel("Angle (deg)")
    plt.ylabel("Normalized Spectrum (dB)")
    plt.grid()
    plt.show()