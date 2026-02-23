import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_conv_layer_names(model):
    """Return a list of Conv2D layer names (in model order)."""
    names = []
    def _collect(layers_list):
        for layer in layers_list:
            if isinstance(layer, tf.keras.layers.Conv2D):
                names.append(layer.name)
            if hasattr(layer, 'layers') and layer.layers:
                _collect(layer.layers)
    _collect(model.layers)
    return names

def compute_heatmap_with_fallback(img_array, model, candidate_layer_names):
    """Try candidate conv layers (last-first) until valid gradients are produced."""
    for layer_name in reversed(candidate_layer_names):
        try:
            heatmap = make_gradcam_heatmap(img_array, model, layer_name)
            return heatmap, layer_name
        except RuntimeError:
            continue
    raise ValueError("None of the candidate conv layers produced valid gradients.")

def make_gradcam_heatmap(img_array, model, conv_layer_name):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    _ = model(img_tensor)  # Build graph

    conv_layer = model.get_layer(conv_layer_name)
    grad_model = tf.keras.Model(
        model.input,
        [conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=True)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients are None — model graph broken.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, output_path, alpha=0.4, colormap=cm.jet):
    # Charger l'image originale en pleine résolution
    img = Image.open(img_path).convert('RGB')
    orig_size = img.size  # (width, height)

    # Redimensionner le heatmap à la taille originale
    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(orig_size, resample=Image.BICUBIC)

    # Appliquer le colormap
    cmap = colormap(np.array(heatmap_resized) / 255.0)  # Normaliser à 0-1
    cmap_img = Image.fromarray((cmap[:, :, :3] * 255).astype('uint8'))

    # Blend sur l'image originale
    overlay = Image.blend(img, cmap_img, alpha=alpha)
    overlay.save(output_path)

def process_image(image_path, model, target_size, conv_candidate_names, out_dir, alpha=0.4):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)

    heatmap, used_layer = compute_heatmap_with_fallback(img_tensor, model, conv_candidate_names)
    out_name = os.path.splitext(os.path.basename(image_path))[0]
    out_file = os.path.join(out_dir, f"{out_name}_gradcam_{used_layer}.png")
    save_and_display_gradcam(image_path, heatmap, out_file, alpha=alpha)
    return heatmap, out_file

def process_folder(folder, model, target_size, conv_candidate_names, out_dir, alpha=0.4, aggregate=False):
    os.makedirs(out_dir, exist_ok=True)
    heatmaps = []
    img_paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not img_paths:
        raise ValueError(f"No images found in {folder}")

    for p in img_paths:
        h, out_file = process_image(p, model, target_size, conv_candidate_names, out_dir, alpha=alpha)
        heatmaps.append(h)

    if aggregate:
        avg = np.mean(np.stack([h for h in heatmaps]), axis=0)
        avg_file = os.path.join(out_dir, os.path.basename(folder.rstrip(os.sep)) + '_avg_gradcam.png')
        save_and_display_gradcam(img_paths[0], avg, avg_file, alpha=alpha)
        return avg, avg_file
    return heatmaps, None

def main():
    parser = argparse.ArgumentParser(description='Grad-CAM visualisation for a Keras model')
    parser.add_argument('--model', default='reseauncnn_best.keras', help='Path to .keras model file')
    parser.add_argument('--image', help='Path to a single image file')
    parser.add_argument('--folder', help='Path to a folder of images')
    parser.add_argument('--out', default='gradcam_outputs', help='Output folder to save overlays')
    parser.add_argument('--size', type=int, nargs=2, default=(224, 224), help='Model input size (width height)')
    parser.add_argument('--alpha', type=float, default=0.4, help='Overlay alpha (0-1)')
    parser.add_argument('--aggregate', action='store_true', help='Aggregate heatmaps across a folder')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    print(f"Loading model from {args.model}...")
    model = tf.keras.models.load_model(args.model)
    _ = model(tf.zeros((1, args.size[1], args.size[0], 3)))  # Build model

    # Utilisation directe de la couche last_conv
    conv_candidates = ["last_conv"]
    print("Using last_conv for Grad-CAM")

    os.makedirs(args.out, exist_ok=True)

    if args.image:
        heatmap, out_file = process_image(args.image, model, tuple(args.size), conv_candidates, args.out, alpha=args.alpha)
        print(f"Saved Grad-CAM overlay to {out_file}")
    elif args.folder:
        if args.aggregate:
            avg, avg_file = process_folder(args.folder, model, tuple(args.size), conv_candidates, args.out, alpha=args.alpha, aggregate=True)
            print(f"Saved aggregated Grad-CAM to {avg_file}")
        else:
            heatmaps, _ = process_folder(args.folder, model, tuple(args.size), conv_candidates, args.out, alpha=args.alpha, aggregate=False)
            print(f"Saved {len(heatmaps)} Grad-CAM overlays in {args.out}")
    else:
        print("Specify --image <path> or --folder <dir> to generate Grad-CAM overlays.")

if __name__ == '__main__':
    main()