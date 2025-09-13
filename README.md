# chromaVive: Deep Learning-Powered Image Colorization

chromaVive is a web application that transforms grayscale images into vibrant, colorized ones. The underlying technology leverages two state-of-the-art deep learning models developed by Richard Zhang and his team at **UC Berkeley**. This README will delve into the research papers behind these models, **ECCV16** and **SIGGRAPH17**, which have pushed the boundaries of automatic and user-guided image colorization.

<div style="text-align: center;">
  <img src="./demo.png" alt="chromaVive" style="width: 100%;">
</div>

## Research Papers and Models

### 1. **Colorful Image Colorization (ECCV16)**

**Paper**: [Colorful Image Colorization](https://arxiv.org/abs/1603.08511) by *Richard Zhang, Phillip Isola, and Alexei A. Efros, ECCV 2016*.

This work presents a fully-automatic approach to colorize grayscale images with vibrant, realistic colors. Traditionally, colorizing black-and-white photos was seen as an under-constrained problem, often resulting in desaturated and dull color outputs. The authors propose a new solution, framing colorization as a **classification task**, rather than regression, which allows the model to generate plausible, rich colorizations by predicting a distribution of potential colors for each pixel.

**Key Features**:
- **Learning from Large-Scale Data**: The model is trained on over a million color images. It learns semantic mappings from grayscale lightness to plausible color distributions, capturing the statistical relationships between textures, objects, and their natural colors.
- **Class-Rebalancing Loss**: To address the challenge of diverse color possibilities, the authors introduce a class-rebalancing loss function, which emphasizes rarer, saturated colors. This prevents the model from defaulting to dull color choices and encourages vibrant results.
- **Perceptual Realism**: The researchers introduce a “colorization Turing test,” wherein human evaluators are asked to differentiate between model-generated and real images. The ECCV16 model successfully fools evaluators 32% of the time, indicating the model’s capability to produce near-photorealistic colorizations.

This paper and model are groundbreaking because they prioritize **plausibility over accuracy** in color choices, resulting in colorizations that are not just technically sound but also aesthetically compelling. The ECCV16 model is ideal for fully automated colorization with minimal need for user intervention.

### 2. **Real-Time User-Guided Image Colorization with Learned Deep Priors (SIGGRAPH17)**

**Paper**: [Real-Time User-Guided Image Colorization with Learned Deep Priors](https://arxiv.org/abs/1705.02999) by *Richard Zhang, Jun-Yan Zhu, Phillip Isola, Xinyang Geng, Angela S. Lin, Tianhe Yu, and Alexei A. Efros, SIGGRAPH 2017*.

This paper extends colorization capabilities by incorporating user interaction into the model, enabling **real-time, user-guided colorization**. While automatic colorization models can produce high-quality color images, some regions in an image may require specific colors for artistic or historical accuracy, which a fully automated model might overlook.

**Key Features**:
- **Interactive Colorization with User Hints**: The model accepts “hint” points, where users can specify colors in certain areas of the image. These hints guide the model to produce a colorization that respects the user’s preferences while maintaining the learned priors from large-scale data.
- **Deep Priors and Fusion of Low- and High-Level Cues**: By combining low-level image features with high-level semantic information, the model achieves a strong balance between realism and control. The learned deep priors make it possible to propagate user-defined colors coherently throughout the image.
- **Suggested Color Palettes**: To enhance user experience, the model provides a data-driven palette with likely colors based on the context of the grayscale input. This helps users intuitively select plausible colors and apply them in real-time, ensuring ease of use even for non-experts.
- **Real-Time Processing**: The model operates in a single feed-forward pass, which allows for real-time responsiveness. This feature is particularly useful for users who want to quickly iterate and experiment with various color schemes.

This user-guided approach represents a shift from fully automatic to semi-automatic colorization, blending the efficiency of deep learning with human creativity and preferences. It’s ideal for applications where historical accuracy or artistic interpretation is important.

## Application in chromaVive

In chromaVive, users can choose between these two models:
- **ECCV16**: For automatic colorization, where the model independently generates colorized versions.
- **SIGGRAPH17**: For user-guided colorization, where users can provide color hints to influence the colorization output interactively.

## Installation and Setup

To use the application, please install the dependencies and set up the pre-trained models as detailed in the following steps.

### Prerequisites
- **Python** 3.7+
- **PyTorch**
- **Flask** and **Flask-Session** for web handling and session management

### Installation
1. **Navigate to the working directory**:
   ```bash
   cd chromaVive
   ```
2. **Install the required Python packages**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Navigate to the `src` directory**:
   ```bash
   cd src
   ```

### Usage
Run the Flask server with:
```bash
flask run
```
Open your browser and navigate to `http://localhost:5000` to begin colorizing images.

## Acknowledgments and Credits

chromaVive is powered by the work of *Richard Zhang, Phillip Isola, Alexei A. Efros*, and others, whose research in deep learning for image colorization has made significant contributions to both computer vision and graphics.

- [ECCV16 Project Page](http://richzhang.github.io/colorization/) by Richard Zhang et al.
- [SIGGRAPH17 Project Page](http://richzhang.github.io/InteractiveColorization/) by Richard Zhang et al.

The original code for these models and additional resources are available on the respective project pages above. All credits for model architecture and core algorithms go to the original authors and their research teams.