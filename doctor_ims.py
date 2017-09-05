from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageOps
import urllib2
import random
import glob
import numpy as np
import cv2
import numpy as np
from scipy import stats
import os

import socket
HOSTNAME = socket.gethostname()

# things we need to load for text insertion
if 'abby' in HOSTNAME:
    fontDir = '/Users/abby/Documents/repos/fonts'
    peopleDir = '/Users/abby/Documents/datasets/people_crops'
else:
    fontDir = '/project/focus/datasets/fonts'
    peopleDir = '/project/focus/datasets/traffickcam/people_crops'

possible_fonts = glob.glob(fontDir+'/*/*/*.ttf')

word_site = "http://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"
response = urllib2.urlopen(word_site)
txt = response.read()
words = txt.splitlines()

# things we need to load for people insertion
people_crops = glob.glob(os.path.join(peopleDir,'*.png'))

def doctor_im(img,ind):
    percent_insta_filters = .4
    percent_rotate = .2
    percent_crop = .5
    percent_text = .1
    percent_people = .5

    im = Image.fromarray(img)
    b, g, r = im.split()
    im = Image.merge("RGB", (r, g, b))
    # crop_size
    print 'crop'
    if random.random() <= percent_crop:
        im = crop_im(im)

    # people
    print 'draw person'
    if random.random() <= percent_people:
        im = draw_person(im)

    # rotate
    print 'rotate'
    if random.random() <= percent_rotate:
        im = rotate_im(im)

    # filter
    print 'filter'
    if random.random() <= percent_insta_filters:
        possible_filters = ['hscb_filter','color_filter']
        whichFilter = random.choice(possible_filters)
        if whichFilter == 'hscb_filter':
            im = Image.fromarray(hscb_filter(np.asarray(im)))
        else:
            im = Image.fromarray(color_filter(np.asarray(im)))

    # text
    print 'text'
    if random.random() <= percent_text:
        draw = ImageDraw.Draw(im)
        word_x_loc = random.choice(range(10,im.size[0]/2))
        word_y_loc = random.choice(range(im.size[1]/2,3*im.size[1]/4))
        phone_x_loc = word_x_loc+random.choice(range(10,100))
        phone_y_loc = word_y_loc+random.choice(range(10,100))
        fontStyle = random.choice(possible_fonts)
        sz1 = int(np.mean(im.size)*.05)
        sz2 = np.min((phone_y_loc-word_y_loc,int(np.mean(im.size)*.1)))
        if sz2 <= sz1:
            fontSize = sz2
        else:
            fontSize = random.choice(range(sz1,sz2))
        font = ImageFont.truetype(fontStyle,fontSize)
        wordStr = random_words()
        phoneNum = random_phone_number()
        textColor = random_color()
        draw = draw_text(draw,wordStr,font,word_x_loc,word_y_loc,textColor)
        draw = draw_text(draw,phoneNum,font,phone_x_loc,phone_y_loc,textColor)

    print 'save back'
    im = im.convert('RGB')
    # im.save('/Users/abby/Desktop/'+str(ind)+'.jpg')
    b, g, r = im.split()
    im = Image.merge("RGB", (b, g, r))

    im = np.array(im)

    return im

## INSTAGRAM FILTERS
# instagram filter code from: https://github.com/weilunzhong/image-filters
def brightness_contrast(im, alpha = 1.0, beta = 0):
    im_contrast = im * (alpha)
    im_bright = im_contrast + (beta)
    # im_bright = im_bright.astype(int)
    im_bright = stats.threshold(im_bright,threshmax=255, newval=255)
    im_bright = stats.threshold(im_bright,threshmin=0, newval=0)
    im_bright = im_bright.astype(np.uint8)
    return im_bright

def channel_enhance(im, channel, level=1):
    im = im.copy()
    if channel == 'B':
        blue_channel = im[:,:,0]
        # blue_channel = (blue_channel - 128) * (level) +128
        blue_channel = blue_channel * level
        blue_channel = stats.threshold(blue_channel,threshmax=255, newval=255)
        im[:,:,0] = blue_channel
    elif channel == 'G':
        green_channel = im[:,:,1]
        # green_channel = (green_channel - 128) * (level) +128
        green_channel = green_channel * level
        green_channel = stats.threshold(green_channel,threshmax=255, newval=255)
        im[:,:,0] = green_channel
    elif channel == 'R':
        red_channel = im[:,:,2]
        # red_channel = (red_channel - 128) * (level) +128
        red_channel = red_channel * level
        red_channel = stats.threshold(red_channel,threshmax=255, newval=255)
        im[:,:,0] = red_channel
    im = im.astype(np.uint8)
    return im

def hue_saturation(im_rgb, alpha = 1, beta = 1):
    im_hsv = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2HSV)
    hue = im_hsv[:,:,0]
    saturation = im_hsv[:,:,1]
    hue = stats.threshold(hue * alpha ,threshmax=179, newval=179)
    saturation = stats.threshold(saturation * beta,threshmax=255, newval=255)
    im_hsv[:,:,0] = hue
    im_hsv[:,:,1] = saturation
    im_transformed = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
    return im_transformed

def hscb_filter(im):
    hue = random.choice(range(95,105))/100.0
    saturation = random.choice(range(90,150))/100.0
    contrast = random.choice(range(90,150))/100.0
    brightness = random.choice(range(-20,100))
    im2 = hue_saturation(im, hue, saturation)
    im2 = brightness_contrast(im2, contrast, brightness)
    return im2

def color_filter(im):
    r_channel = random.choice(range(90,120))/100.0
    g_channel = random.choice(range(90,120))/100.0
    b_channel = random.choice(range(90,120))/100.0
    im2 = channel_enhance(im, "R", r_channel)
    im2 = channel_enhance(im, "G", g_channel)
    im2 = channel_enhance(im, "B", b_channel)
    return im2

## ROTATION
def rotate_im(im):
    alpha = im.convert('RGBA')
    angle = random.choice(range(-30,30))
    rot = alpha.rotate(angle, resample=Image.BILINEAR)
    pixels = np.asarray(rot)
    solid_pixels = np.where(pixels[:,:,3]>0)
    possible_top_corners = []
    for iy in range(im.size[1]/2):
        for ix in range(im.size[0]):
            if pixels[iy,ix][3] > 0:
                possible_top_corners.append((iy,ix))
                break
    possible_bottom_corners = []
    for iy in range(im.size[1]-1,im.size[1]/2,-1):
        for ix in range(im.size[0]-1,im.size[0]/2,-1):
            if pixels[iy,ix][3] > 0:
                possible_bottom_corners.append((iy,ix))
                break
    best_bounds = []
    best_area = 0
    for iy,ix in possible_top_corners:
        for ay, ax in possible_bottom_corners:
            if pixels[iy,ax][3] > 0 and pixels[ay,ix][3] > 0:
                area = (ay-iy)*(ax-ix)
                if area > best_area:
                    best_area = area
                    best_bounds = (ix,iy,ax,ay)
    cropped_rot = rot.crop((best_bounds[0],best_bounds[1],best_bounds[2],best_bounds[3]))
    if best_bounds[2]-best_bounds[0] > best_bounds[3]-best_bounds[1]:
        # landscape image, crop to 4:3
        height = cropped_rot.size[1]
        new_width = 4.0/3.0*height
        center_x = (best_bounds[2]-best_bounds[0])/2
        new_x1 = center_x-new_width/2
        new_x2 = center_x+new_width/2
        cropped_rot = cropped_rot.crop((new_x1,0,new_x2,cropped_rot.size[1]))
    else:
        # portrait image, crop to 3:4
        width = cropped_rot.size[0]
        new_height = 4.0/3.0*width
        center_y = (best_bounds[3]-best_bounds[1])/2
        new_y1 = center_y-new_height/2
        new_y2 = center_y+new_height/2
        cropped_rot = cropped_rot.crop((0,new_y1,cropped_rot.size[0],new_y2))
    if cropped_rot.size[0]*cropped_rot.size[1] < 50000:
        cropped_rot = im
    return cropped_rot

def crop_im(im):
    crop_ratio = float(3)/float(5)
    width = im.size[0]
    height = im.size[1]
    possible_x = range(0,int(width-width*crop_ratio))
    possible_y = range(0,int(height-height*crop_ratio))
    new_start_x = random.choice(possible_x)
    new_start_y = random.choice(possible_y)
    new_end_x = new_start_x + int(width*crop_ratio)
    new_end_y = new_start_y + int(height*crop_ratio)
    im2 = im.crop((new_start_x,new_start_y,new_end_x,new_end_y))
    if im2.size[0]*im2.size[1] < 50000:
        im2 = im
    return im2

def random_words():
    randWords = random.sample(words,2)
    randWordStr = '%s %s' % (randWords[0],randWords[1])
    if random.random() > .5:
        randWordStr = randWordStr.title()
    return randWordStr

def random_phone_number():
    n = '0000000000'
    while '9' in n[3:6] or n[3:6]=='000' or n[6]==n[7]==n[8]==n[9]:
        n = str(random.randint(10**9, 10**10-1))
    return n[:3] + '-' + n[3:6] + '-' + n[6:]

def random_color():
    rgbl=[random.choice((0,128,255)),random.choice((0,128,255)),random.choice((0,128,255))]
    random.shuffle(rgbl)
    return tuple(rgbl)

def draw_text(draw,text,font,x,y,textColor):
    # thicker border
    draw.text((x-1, y-1), text, font=font, fill=(255,255,255))
    draw.text((x+1, y-1), text, font=font, fill=(255,255,255))
    draw.text((x-1, y+1), text, font=font, fill=(255,255,255))
    draw.text((x+1, y+1), text, font=font, fill=(255,255,255))
    # now draw the text over it
    draw.text((x, y), text, font=font, fill=(textColor[0],textColor[1],textColor[2]))
    return draw

# people
def draw_person(im):
    try: # ugh, fix in the future, just trying to get it running right now
        im2 = im.convert('RGBA')
        person = Image.open(random.choice(people_crops)).convert('RGBA')
        if random.random() > .5:
            person = ImageOps.mirror(person)
        angle = random.choice(range(-30,30))
        rotated_person = person.rotate(angle,resample=Image.BILINEAR, expand=1)
        y_offset = random.choice(range(0,int(im2.height*.2)))
        new_height = random.choice(range(int(im2.height*.7),im.height-y_offset))
        new_width = (im2.width/im2.height)*new_height
        x_offset = random.choice(range(int(im2.width*.1),int((im2.width-new_width)*.9)))
        new_person = rotated_person.resize((new_width,new_height), Image.ANTIALIAS)
        im2.paste(new_person, (x_offset, y_offset), new_person)
        return im2
    except:
        return im
