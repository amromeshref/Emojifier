import sys
import os

# Add the repository directory path to the Python path
REPO_DIR_PATH = os.path.normpath(os.path.join(
    os.path.join(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(REPO_DIR_PATH)


from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.uix.anchorlayout import AnchorLayout
from src.predict_model import ModelPredictor

# Set the background color of the window to white
Window.clearcolor = (1, 1, 1, 1)

# Initialize the predictor
predictor = ModelPredictor()

class EmojifierApp(App):
    def build(self):
        # Define the main layout
        anchor_layout = AnchorLayout()
        layout = BoxLayout(orientation='vertical', padding=20, size_hint=(None, None), width=800, height=300)
        
        # Add widgets to the layout
        label = Label(text="Enter a sentence:", size_hint=(1, None), height=50, color=(0, 0, 0, 1))
        layout.add_widget(label)

        self.input_box = TextInput(multiline=False, size_hint=(1, None), height=50)
        layout.add_widget(self.input_box)

        button = Button(text="Get the Emoji", size_hint=(1, None), height=50, on_press=self.on_button_click)
        layout.add_widget(button)

        # Output Image widget
        self.output_image = Image(source="", size_hint=(1, None), height=100)
        layout.add_widget(self.output_image)

        # Center the BoxLayout in the AnchorLayout
        anchor_layout.add_widget(layout)
        return anchor_layout

    def on_button_click(self, instance):
        # Get the sentence from the TextInput
        sentence = self.input_box.text

        # Process the sentence using your model
        processed_output = predictor.predict(sentence)

        # Set the image source based on processed output
        if processed_output == 0:
            self.output_image.source = "images/heart.jpg"  # Replace with actual path to your heart image
        elif processed_output == 1:
            self.output_image.source = "images/baseball.jpg"  # Replace with actual path to your baseball image
        elif processed_output == 2:
            self.output_image.source = "images/smile.jpg"  # Replace with actual path to your smile image
        elif processed_output == 3:
            self.output_image.source = "images/sad.jpg"  # Replace with actual path to your sad image
        elif processed_output == 4:
            self.output_image.source = "images/fork.jpg"  # Replace with actual path to your fork image
        else:
            self.output_image.source = "images/question.jpg"  # Default image for unknown output

if __name__ == '__main__':
    EmojifierApp().run()

