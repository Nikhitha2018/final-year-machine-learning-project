from django import forms

# Creating Forms BlackForm  to take user input for our Django App
#importing required models to create forms 
from .models import Black,UserRegister
class BlackForm(forms.ModelForm):
    class Meta:
        model=Black
        fields='__all__'
class UserRegisterForm(forms.ModelForm):
    class Meta:
        model=UserRegister
        fields='__all__'