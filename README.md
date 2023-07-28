# Emojify---Emotion-Classification-using-Facial-Landmark-Detection

Machine Learning / Data Mining Combined project
(Team of 4)

Used various Machine Learning techniques to classify the emotion of a person in a labeled dataset of 48x48 black and white headshots.

Our process began by using Dlib to extract 68 facial landmarks (points on the face that define the outline of important features such as eyes, lips, eyebrows, outline of head, etc.)
We then constructed feature vectors using L2 distances between specific pairs of points
Finally we used these feature vectors to create our feature matrix, which was then analyzed using SVM, Random Forest, LightGBM, AdaBoosting, Khan-NN, and CNN, and DeepFace: VGG-Face (for evaluation)

Our results using ML methods in addition to our CNN were comparable to the industry standard evaluation model VGG-Face
