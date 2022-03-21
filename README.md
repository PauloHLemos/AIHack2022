# AIHack2022
A model to predict microfluidic drop interactions.

Top 3 project on the drop coalescence challenge.

### Problem
Given the first 100 frames, predict whether two silver droplets will coalesce or not.

### Data
The data was given in the form of around 500 frames of matrices using a binary encoding to represent the outline of colliding the droplets. 
The first step was to remove the noise from the input. This was done using open-cv to detect the contour of the main shapes and disregard any shapes with insufficient area to be a valid droplet.

#### Unfiltered data:
![blob_unfiltered](https://user-images.githubusercontent.com/58424964/159328056-1918aa20-dded-493d-8735-43c67de67dd1.gif)
#### Filtered data:
![blob_filtered](https://user-images.githubusercontent.com/58424964/159329695-9afc2073-5314-455f-b71d-415510f4e0a0.gif)


### Approach
The approach we went for was to extract the significant parameters which determined whether two droplets coalesce or not.
These were size, momentum and shape of the droplet. Mass is also a factor to consider, but since the samples were all of the same substance silver, the area of the droplet served as a good indicator for the relative mass among test and train data.

Using the open-cv library also gave us the advantage that the area and center of the shape could be easily obtained. We used the center to calculate the general movement of the drop relative to the previous frame and used this to calculate the velovity of the droplet.

We then fed these parameters into a 6-layered neural network and achieved a 70% accuracy.
