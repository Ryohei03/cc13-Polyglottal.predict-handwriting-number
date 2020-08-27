from django.db import models

def get_image_path(self, filename):
    path = 'images/'
    name = "tmp.png"
    return path + name

class Image(models.Model):
    picture = models.ImageField(upload_to=get_image_path)

    def __str__(self):
        return self.title

