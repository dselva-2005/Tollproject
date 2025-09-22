from django.db import models

# Create your models here.
class Plate(models.Model):
    plate_no = models.CharField(max_length=20)