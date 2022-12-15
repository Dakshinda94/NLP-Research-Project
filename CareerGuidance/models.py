from django.db import models


# Create your models here.
class Employee(models.Model):
    Name = models.CharField(max_length=100  )
    Address = models.CharField(max_length=100, blank=True)
    Contact  = models.CharField(max_length=12 )
    Email  = models.CharField(max_length=12 )
    Password  = models.CharField(max_length=12 )

    def __str__(self):
        return self.Email


# Create your models here.
class jobprofile(models.Model):
    Email = models.CharField(max_length=100  )
    skill = models.CharField(max_length=100, blank=True)
    experience  = models.CharField(max_length=100 )
    jobIndustry  = models.CharField(max_length=100 )
    def __str__(self):
        return self.Email




# Create your models here.
class Student(models.Model):
    StudentName = models.CharField(max_length=100  )
    Address = models.CharField(max_length=100, blank=True)
    Interest = models.CharField(max_length=100, blank=True)
    Email  = models.CharField(max_length=12 )
    Password  = models.CharField(max_length=12 )

    def __str__(self):
        return self.Email

# Create your models here.
class Student_jobprofile(models.Model):
    Email = models.CharField(max_length=100  )
    skill = models.CharField(max_length=100, blank=True)
    experience  = models.CharField(max_length=100 )
    def __str__(self):
        return self.Email
