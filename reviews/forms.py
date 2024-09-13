from django import forms

class ReviewForm(forms.Form):
    review = forms.CharField(label='Review', widget=forms.Textarea(attrs={'rows': 5, 'cols': 40}))
