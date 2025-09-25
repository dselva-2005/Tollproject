from django.db import models
from django.contrib.auth.models import User
import uuid

class DetectedPlates(models.Model):
    plate_no = models.CharField(max_length=20)


class CarExit(models.Model):
    exit_id = models.UUIDField(primary_key=True,default=uuid.uuid4,editable=False)
    distance_from_entry = models.IntegerField()

class Rfidcard(models.Model):
    id = models.CharField(primary_key=True,unique=True, max_length=20)
    car_number = models.CharField(max_length=20)
    card_number = models.CharField(max_length=20)

class Reader(models.Model):
    id = models.CharField(primary_key=True)
    car_exit = models.OneToOneField(CarExit,on_delete=models.CASCADE)