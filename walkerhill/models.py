from django.db import models

class CrawledText(models.Model):
    # 텍스트의 제목을 저장 
    title = models.CharField(max_length=255, blank=True, null=True)  
    # 크롤링된 텍스트를 저장
    content = models.TextField()  

    def __str__(self):
        return self.title if self.title else "No Title"