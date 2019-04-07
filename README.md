# neural-style-transfer
neural style transfer

## INSTRUCTIONS
if using google colab first import nst_utils
first take out mask by using mask-rcnn module and passing content image to the module which will return mask image.https://www.dropbox.com/s/5pzxmero2rbvmv3/frozen_inference_graph.pb?dl=0
# NEURAL STYLE TRANSFER - VOID HACK()- BY ML CHAMPS (11)
  
  
  APPROACH :-
    1) Used VGG-19 model trained on face(person) detection dataset.
    2) Used mask rcnn implementation for taking out masks from images(h).
    3) then passed that mask image to preserve content of the image .
    4) then our implementation used 3 losses in place of tradional 2 losses
    5) we calculate content loss ,Style loss (foreground image) and style loss (background image)
    6) we minimise the total of all this 3 losses to optimize and preserve the main content of image.
    7) we have followed standard guidance followed by research paper to train background loss using earlier layer and content loss using upper layer which preserve image details.
    8) going through 300 iterations our model performs really well.

# COLOR PRESERVATION

    color preservation technique is used to preserve colour of the content image and applying that colour scheme to the generated image , we have used image histogram matching approach to match the color pattern of content image with genralizeed image
