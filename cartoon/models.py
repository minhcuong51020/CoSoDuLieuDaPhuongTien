from django.db import models

# Create your models here.


class ImageCartoon(models.Model):
    img = models.ImageField(upload_to='uploads/')