#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Rodrigo Agundez'
SITENAME = 'Rodrigo Agundez'
SITEDESCRIPTION = ''
SITEURL = ''

# plugins
PLUGIN_PATHS = ['plugins']
PLUGINS = ['i18n_subsites', 'tipue_search']
JINJA_ENVIRONMENT = {'extensions': ['jinja2.ext.i18n']}

# theme and theme localization
THEME = 'theme'
TIMEZONE = 'Europe/Amsterdam'
DEFAULT_DATE_FORMAT = '%a, %d %b %Y'
DEFAULT_LANG = 'en'
LOCALE = 'en_US'

# content paths
PATH = 'content'
PAGE_PATHS = ['pages']
ARTICLE_PATHS = ['blog', 'timeline']

# logo path, needs to be stored in PATH Setting
LOGO = '/images/logo.svg'

# special content
HERO = [
    {
        'image': '/images/hero/background-1.jpg',
        # for multilanguage support, create a simple dict
        'title': 'Some special content',
        'text': 'Any special content you want to tease here',
        'links': [{
            'icon': 'icon-code',
            'url': 'https://github.com/claudio-walser/pelican-fh5co-marble',
            'text': 'Github'
        }]
    }, {
        'image': '/images/hero/background-2.jpg',
        # keep it a string if you dont need multiple languages
        'title': 'Uh, special too',
        # keep it a string if you dont need multiple languages
        'text': 'Keep hero.text and hero.title a string if you dont need multilanguage.',
        'links': []
    }, {
        'image': '/images/hero/background-3.jpg',
        'title': 'No Blogroll yet',
        'text': 'Because of space issues in the man-nav, i didnt implemented Blogroll links yet.',
        'links': []
    }, {
        'image': '/images/hero/background-4.jpg',
        'title': 'Ads missing as well',
        'text': 'And since i hate any ads, this is not implemented as well',
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
    'mail': 'info@rragundez.io',
    # keep it a string if you dont need multiple languages
    'text': 'Learn more about the creator of this theme or just drop a message.',
    'link': 'contact.html',
    # the address is also taken for google maps
    'address': 'Amsterdam, The Netherlands',
    'phone': '+31-mexicano'
}

# navigation and homepage options
DISPLAY_PAGES_ON_MENU = True
DISPLAY_PAGES_ON_HOME = False
DISPLAY_CATEGORIES_ON_MENU = False
DISPLAY_TAGS_ON_MENU = False
USE_FOLDER_AS_CATEGORY = True
PAGE_ORDER_BY = 'order'

MENUITEMS = [
    ('Timeline', 'timeline.html'),
    ('Blog', 'blog.html'),
    ('Categories', 'categories.html'),
    ('Tags', 'tags.html'),
    ('Contact', 'contact.html')
]

DIRECT_TEMPLATES = [
    'index',
    'timeline',
    'blog',
    'categories',
    'tags',
    'authors',
    'search',  # needed for tipue_search plugin
    'contact'  # needed for the contact form
]

# setup disqus
DISQUS_SHORTNAME = 'gitcd-dev'
DISQUS_ON_PAGES = False  # if true its just displayed on every static page, like this you can still enable it per page

# setup google maps
GOOGLE_MAPS_KEY = 'AIzaSyCefOgb1ZWqYtj7raVSmN4PL2WkTrc-KyA'
