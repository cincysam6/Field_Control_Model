# Field_Control_Model
This Repo hosts an NFL Field Control Model. Here is a quick overview of the Repo structure:

1) Data folder hosts an individual play I use in my example notebook to test functionality
2) src folder will hold all our functions and can be used similar to a package once you clone this repo into your environment
3) Field_Control_Fuctions notebook should be deprecated at some point. This is old and not necessary for the package to be used
4) example_usage should be your starting point. This can be pulled into colab and you can run through the notebook end to end to see the various function usage
5) interactive_field_control allows you to play with parameters and print those paramaters to see how changes effect the shape at various speeds. Good for finding hyperparameters 
6) influence_class_dev is a sandbox to play with the model and change it and evaluate impact

8) interactive_parameter_selection is a WIP but it takes the same concept as interactive_field_control and bakes it into a shiny app. 
