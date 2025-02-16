# Projekt Bild- und Signalverarbeitung - Kamerakalibrierung und Vermessung von Paketen

Im Kontext des Moduls Bild- und Signalverarbeitung folgt die Studienarbeit von Adrian Schiel (7022935) und Mirko Labitzke (7021691). Betreut wird diese Arbeit von Prof. Koch. <br>
Die Aufgabenstellung ist dabei, selber eine Kamera in Form einer Webcam oder Ähnlichem zu kalibrieren und mithilfe der Kalibrierung Pakete zu vermessen.

### Zielsetzung

Im Rahmen der Studienarbeit ist das Ziel, eine Webcam, in unserem Fall eine "Logitech c925e" 1080p ohne Auto-Fokus, zu kalibrieren. Dazu werden innerhalb einer Fotobox verschiedene Bilder von Schachbrettmustern gemacht und anschließend diese mithilfe von Algorithmen aus der OpenCV Bibliothek genutzt, um den Fehler und andere Werte der Kamera zu ermitteln. Die Fotobox wird verwendet, um besonders gute Bilder mit kontrollierten Lichtverhältnissen und wenig Einfluss von außen machen zu können. <br>
Mithilfe der Kamerakalibrierung sollen verschieden große und farbige Pakete erkannt und vermessen werden. Die Bilder der Pakete werden ebenfalls unter genau den gleichen Bedingungen gemacht. Bei der Vermessung der Pakete wird sich erst auf die 2D Vermessung von Höhe und Breite konzentriert. Die 3D Vermessung von Höhe, Breite und Tiefe wäre der Schritt, der darauf folgt, wenn die 2D Vermessung gut klappt.  <br>
Die Zielsetzung allgemein ist bei dem Projekt einfach zu definieren:
#### Es soll eine möglichst gute Kamerakalibrierung gefunden werden, mit welcher die unterschiedlichen Pakete möglichst genau vermessen werden können!