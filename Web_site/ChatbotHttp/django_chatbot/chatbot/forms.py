from django import forms
from django.contrib.admin.widgets import AutocompleteSelect
from django.contrib import admin
from .models import Audios
from django.core.exceptions import ValidationError
from captcha.fields import ReCaptchaField


class User_nameForm(forms.Form):
    user_name = forms.CharField(
                                widget=forms.TextInput(
                                    attrs={'class': 'form-control',
                                        'placeholder':'Атыңызды жазыңыз',
                                           "style": "width:200px"}))
    user_name.widget.attrs.update(size="40")
class MyModel(forms.ModelForm):
    class Meta:
        fields = ('text',)
        widgets = {
            'text': AutocompleteSelect(
                Audios.audio_file.field.remote_field,
                admin.site,
                attrs={'style':'width: 100px'}
            ),
        }



class TextForm(forms.ModelForm):

    #text = forms.CharField(label='Enter Text', widget=forms.Textarea(attrs={'class': 'form-control'}), required=False)
    class Meta:
        model = Audios
        fields = ['text']
        widgets = {
            'text': forms.Textarea(attrs={ "style": "height:200px; resize: vertical; overflow-y: auto; width:80%",'class': 'form-control', 'required':'required'})

        }
    #captcha = ReCaptchaField(label='')
    def clean_text(self):
        text = str(self.cleaned_data['text'])
        if len(text) > 5000:
            raise ValidationError('Кайра жазыңыз')
        return text
    def __init__(self, *args, **kwargs):
        super(TextForm, self).__init__(*args, **kwargs)
        self.fields['text'].label = ""

    # def clean(self):
    #     cleaned_data = super().clean()
    #     if not cleaned_data.get('text'):
    #         raise forms.ValidationError('Please enter text ')

# class DocumentForm(forms.ModelForm):
#     captcha2 = ReCaptchaField(label='')
#     class Meta:
#         model = Audios
#         fields = ['audio_file']
#     def __init__(self, *args, **kwargs):
#         super(DocumentForm, self).__init__(*args, **kwargs)
#         self.fields['audio_file'].label = ""
