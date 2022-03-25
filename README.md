# DaVIS
 DaVIS is a Data Visualiser Tool for visualising data in various unconventional methods to provide more insights

# GUI
[![https://infinityjoker-apps.herokuapp.com/](https://pyheroku-badge.herokuapp.com/?app=infinityjoker-apps&style=plastic)](https://infinityjoker-apps.herokuapp.com/)

 - GUI built using streamlit
 - To use app locally,
    - Clone the repo and run [StartUI.sh](StartUI.sh) to view the app on your browser!
 - App is also hosted remotely on heroku using my common host app,
    - [https://infinityjoker-apps.herokuapp.com/](https://infinityjoker-apps.herokuapp.com/)

    - In the Common Host App, simply choose a project to load and click load and deploy.

    - Then go ahead and use the app! :)

    - If you want to change to another app, simply click on View Other Projects in top left and choose any other project and load and deploy.

# Visualisations
## AudioVis
### Audio Spectrogram Visualiser
   - Plots spectrograms of audio files
   
   ![Audio Wave](AudioVis/GeneratedVisualisations/AudioSpectrogram_AudioWave.png)
   ![Audio Spectrogram](AudioVis/GeneratedVisualisations/AudioSpectrogram_Spectrogram.png)
   
## ImageVis
### Image Point Effects
   - Generating Fun 3D Effects from Images
   
   ![Input Image](ImageVis/GeneratedVisualisations/Pika.jpg)
   ![Effect GIF](ImageVis/GeneratedVisualisations/Pika_IPEffect.gif)

### Image 3D Model Generator
   - Generating 3D models from images using depth based on various factors
   
      - GrayScale Depth

         ![Input Image](ImageVis/GeneratedVisualisations/Image3DVis_GrayScaleDepth_InputImage.png)

         ![Depth Image](ImageVis/GeneratedVisualisations/Image3DVis_GrayScaleDepth_DepthImage.png)

         ![Depth Model](ImageVis/GeneratedVisualisations/Image3DVis_GrayScaleDepth_Model.PNG)

      - AI Depth

         ![Input Image](ImageVis/GeneratedVisualisations/Image3DVis_AIDepth_InputImage.PNG)

         ![Depth Image](ImageVis/GeneratedVisualisations/Image3DVis_AIDepth_DepthImage.png)

         ![Depth Model](ImageVis/GeneratedVisualisations/Image3DVis_AIDepth_Model.PNG)

### Image 2D Visualiser
   - Generating Fun Visualisations of Images

      - Split RGB Channels in images
   
      ![RGB Split](ImageVis/GeneratedVisualisations/Image2DVis_RGBChannelSplit.png)

      - View dominant colours only in a image
   
      ![Dominant View](ImageVis/GeneratedVisualisations/Image2DVis_DominantChannel.png)

      ![Dominant Split](ImageVis/GeneratedVisualisations/Image2DVis_RGBDominantChannelSplit.png)

      - Histogram Enhance Images
   
      ![Histogram Enhanced](ImageVis/GeneratedVisualisations/Image2DVis_ImageEnhance.png)

      - View Image Histograms
   
      ![Histogram](ImageVis/GeneratedVisualisations/Image2DVis_ImageHistogram.png)

## AlgoVis

 - Moved to a separate repo: [AlgoVis](https://www.github.com/KausikN/AlgoVis)