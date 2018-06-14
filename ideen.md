# ideen

## Allgemein

* gröbere Klassen als die 10 von der WMO (world meteorological orgnization)
    * z.B. (s. [Wikipedia](https://en.wikipedia.org/wiki/List_of_cloud_types#Cloud_identification_and_classification:_Order_of_listed_types)):
        * stratiform: Cirrostratus, Altostratus, Nimbostratus, Stratus
        * cirriform: Cirrus
        * stratocumuliform: Cirrocumulus, Altocumulus, Stratocumulus
        * cumuliform: Cumulus, Cumulonimbus (eig. cumulonimbiform, aber...)
* Bilder bei der Klassifizierung auf eine Größe bringen
    * Panorama-Bilder sowieso aufsplitten (TODO: Lukas)
    * Bilder kacheln?
        * Bei Klassifizierung: Konfidenzen aufsummieren? 
    * Bilder auf eine Größe bringen...
    * alles lokal --> Originalbilder nicht verändern
* eine Prozedur, die die Bilder lädt und auch die Label zurückgibt
    * Bilder schon vorverarbeitet von Numpy im `temp`-Ordner speichern lassen (mit `numpy.savez_compressed`)
* Befragung von Menschen


## Deep Learning

* Wir haben nicht so viele Bilder...
    * vortrainiertes Netz?
        * Nur Klassifikation neu trainieren?
        * Oder ganzes Netz "tunen"?
        * gibts bei Keras eingebaut
    * Data Augmentation
* eigenes Netz trotzdem einmal versuchen...


## "Klassisch"

* Automatisiertes "Rausschmeißen" von Trainingsbildern?
* kriegen wir es hin, den Himmel "auszuschneiden"?
    * binarisiert per Region Growing???
* Features
    * Mittelwert
    * Standardabweichung
    * Frequenz
    * Histogramme (nach Binarisierung)
    * Zusammenhangskomponenten von Wolken
        * Davon die Frequenz?
    * Template Matching??
    * Kantenerkennung, Richtungshistogramm?????????????
    * Alles nochmal mit Tiling...
* Klassifikation
    * Entscheidungsbaum
        * eventuell automatisch erstellt?
    * k-nearest Neighbors
        * dabei muss der Parameterraum (Gewichtung etc) erforscht werden
        * wie viele Nachbarn??
    * dense neural network
