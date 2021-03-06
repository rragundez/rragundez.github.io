#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Rodrigo Agundez'
SITENAME = 'Rodrigo Agundez'
SITEDESCRIPTION = ''
SITEURL = ''

# plugins
PLUGIN_PATHS = ['plugins']
PLUGINS = ['render_math', 'i18n_subsites', 'tipue_search']
JINJA_ENVIRONMENT = {'extensions': ['jinja2.ext.i18n']}

# theme and theme localization
THEME = 'theme'
TIMEZONE = 'Europe/Amsterdam'
DEFAULT_DATE_FORMAT = '%d %b %Y'
DEFAULT_LANG = 'en'
LOCALE = 'en_US'
STATIC_PATHS = ['images', 'CNAME', 'favicon.ico', 'documents']

# content paths
PATH = 'content'
PAGE_PATHS = ['pages']
ARTICLE_PATHS = ['blog', 'timeline']

LOGO = '/images/logo.svg'

# special content
HERO = [
    {
        'image': '/images/hero/spark_ai.jpg',
        'title': 'Maverick',
        'text': '"An unorthodox or independent-minded person"',
        'links': []
    }, {
        'image': '/images/hero/football.jpg',
        'title': 'Sporter',
        'text': '',
        'links': []
    }, {
        'image': '/images/hero/restart_network.jpg',
        'title': 'Leader',
        'text': '',
        'links': []
    }, {
        'image': '/images/hero/ddsw.jpg',
        'title': 'Knowledge Sharing',
        'text': '',
        'links': []
    }, {
        'image': '/images/hero/nspire.jpg',
        'title': 'Teamplayer',
        'text': '',
        'links': []
    }
]

# Social widget
SOCIAL = (
    ('Github', 'https://www.github.com/rragundez'),
    ('Twitter', 'https://www.twitter.com/rragundez'),
    ('LinkedIn', 'https://www.linkedin.com/in/rodrigo-agundez-2b727258')
)

ABOUT = {
    'image': '/images/about/about.jpeg',
    'mail': 'rragundez@gmail.com',
    # keep it a string if you dont need multiple languages
    'text': 'Just drop a message.',
    'link': 'contact.html',
    # the address is also taken for google maps
    'address': 'Amsterdam, The Netherlands',
    #'phone': '+31-mexicano'
}

# navigation and homepage options
DISPLAY_PAGES_ON_MENU = True
DISPLAY_PAGES_ON_HOME = False
DISPLAY_CATEGORIES_ON_MENU = False
DISPLAY_TAGS_ON_MENU = False
USE_FOLDER_AS_CATEGORY = True
PAGE_ORDER_BY = 'order'

MENUITEMS = [
    ('Rod360°', 'rod360.html'),
    ('Timeline', 'timeline.html'),
    ('Blog', 'blog.html'),
    ('Categories', 'categories.html'),
    ('Tags', 'tags.html'),
    ('Contact', 'contact.html')
]

DIRECT_TEMPLATES = [
    'index',
    'rod360',
    'timeline',
    'blog',
    'categories',
    'tags',
    'search',  # needed for tipue_search plugin
    'contact'  # needed for the contact form
]

# setup disqus
DISQUS_SHORTNAME = 'gitcd-dev'
DISQUS_ON_PAGES = False  # if true its just displayed on every static page, like this you can still enable it per page

# setup google maps
GOOGLE_MAPS_KEY = 'AIzaSyDtLsg5TViozEWlHg4RihNiuUv9T8IPm90'
