from django.contrib import admin
from .models import CrawledText

@admin.register(CrawledText)
class CrawledTextAdmin(admin.ModelAdmin):
    list_display = ('title', 'content')