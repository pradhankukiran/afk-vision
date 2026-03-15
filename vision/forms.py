from django import forms

from .models import Project


class ProjectForm(forms.ModelForm):
    class Meta:
        model = Project
        fields = ["name", "description"]
        widgets = {
            "name": forms.TextInput(
                attrs={
                    "placeholder": "North Block Weed Scout",
                    "autocomplete": "off",
                }
            ),
            "description": forms.Textarea(
                attrs={
                    "rows": 4,
                    "placeholder": "Optional crop notes, weed pressure, or scouting instructions.",
                }
            ),
        }
        labels = {
            "name": "Scouting program name",
            "description": "Scouting notes",
        }


class UploadImageForm(forms.Form):
    image = forms.ImageField(
        label="Crop or field image",
        widget=forms.ClearableFileInput(attrs={"accept": "image/*", "data-file-input": "true"}),
    )
