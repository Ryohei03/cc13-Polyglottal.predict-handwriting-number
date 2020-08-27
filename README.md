# predict-handwriting-number
 (NOTE:“This was created during my time as a student at Code Chrysalis”)

## Overview

`predict-handwriting-number` 
 When an earthquake hits, we use this map for knowing nearby a water station. 
 Also, we may not deeply understand this place and it may be difficult that people find it.
 So when we choose the place, we can look at the picture of the place.

![upload](https://user-images.githubusercontent.com/65406188/91422017-97a07a00-e891-11ea-8592-56e558f7a4e7.png)
![showall](https://user-images.githubusercontent.com/65406188/91422098-ae46d100-e891-11ea-900f-a8f00fa2a513.png)


## Installation
The environment is assumed that "node.js" and "postgreSQL" have already been installed.

You can download a library.

To initialize your environment:

    $ yarn
    $ yarn build
    $ yarn start
    $ node seeds/import.js



## Getting Started

    $ yarn start
    Then open http://localhost:9000/ wtih brouser.

Now the Express server has started and is ready to receive requests.

## Features
Assuming water is needed, this map highlights rivers, ponds, and seas in blue.

Geolocation is used because of searching for current locations.
