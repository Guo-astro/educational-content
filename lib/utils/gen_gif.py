#############################################
# Utility Function: Auto-adjust GIF Parameters
#############################################
import os
import tempfile


def auto_save_gif(ani, filename, target_size=5 * 1024 * 1024,
                  fps=20, initial_dpi=80,
                  min_fps=10, min_dpi=40, dpi_step=5):
    """
    Automatically search for the highest quality (defined here as the product fps*dpi)
    parameters that produce a GIF file smaller than target_size bytes.
    Saves the GIF to the given filename.
    """
    best_quality = 0
    best_params = None
    best_file_size = None

    for dpi in range(initial_dpi, min_dpi - 1, -dpi_step):
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as tmp:
                temp_filename = tmp.name
            try:
                ani.save(temp_filename, writer='pillow', fps=fps, dpi=dpi)
                size = os.path.getsize(temp_filename)
                print(f"Tested fps={fps}, dpi={dpi}: file size = {size / 1024:.2f} KB")
                if size <= target_size:
                    quality = fps * dpi  # Define quality as product of fps and dpi
                    best_quality = quality
                    best_params = (fps, dpi)
                    best_file_size = size
                    break
            except Exception as e:
                print(f"Error with fps={fps}, dpi={dpi}: {e}")
            finally:
                try:
                    os.remove(temp_filename)
                except Exception:
                    pass

    if best_params is not None:
        best_fps, best_dpi = best_params
        print(f"\nSelected parameters: fps={best_fps}, dpi={best_dpi} with file size {best_file_size / 1024:.2f} KB")
        ani.save(filename, writer='pillow', fps=best_fps, dpi=best_dpi)
        print(f"GIF saved as '{filename}'")
    else:
        print("Could not find parameters that yield a file size under the target.")
