"""

Jedes Plugin hat eine eigene Struktur und kann beliebige Dateien und Unterordner enthalten.
Das __init__.py ist der Einstiegspunkt des Plugins. 
    -> ruft die Hauptlogik aus anderen Dateien auf

    Bsp.:

from .clustering import _

def run():
    _ = _()
    
"""

#from .IA_ViT import IA_ViTransformer TODO