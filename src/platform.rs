//! Thin shims over facilities that exist natively but have no browser
//! equivalent. Keeping the `#[cfg]` churn here means the rest of the tree
//! calls one function instead of repeating target gates at every site.

/// The user's home directory, falling back to the working directory.
///
/// On wasm there is no filesystem and no home directory, so this is always
/// `.`. Callers use it purely as a starting point for path construction; the
/// reads and writes themselves compile on web but always return `Err`.
pub fn home_dir() -> std::path::PathBuf {
    #[cfg(not(target_arch = "wasm32"))]
    {
        dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."))
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::path::PathBuf::from(".")
    }
}

/// Whether temporal anti-aliasing can be used on this build.
///
/// False on web. Bevy's TAA shader samples one texture through two different
/// samplers, which WGSL allows but GLSL ES 3.0 cannot express — it uses
/// combined image-samplers. naga fails translation with "A image was used with
/// multiple samplers", and wgpu treats that as fatal, so merely inserting
/// `TemporalAntiAliasing` on a camera panics the whole app on the first frame
/// it builds the pipeline. Callers must fall back to MSAA.
// Unused until the settings screen offers an anti-aliasing choice; the analysis
// in the doc comment is the reason it exists and is worth not rediscovering.
#[allow(dead_code)]
pub const fn taa_supported() -> bool {
    !cfg!(target_arch = "wasm32")
}
