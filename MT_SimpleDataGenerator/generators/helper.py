import random

STREET_NAMES = ['Kaiserstraße', 'Frankfurter Straße', 'Münchner Straße', 'Bachstraße', 'Bahnhofstraße',
                'Mainzer Landstraße', 'Rossertstraße', 'Am Hubland', 'Am Galgenberg', 'Emil-Fischer-Straße',
                'Sanderring', 'Roßmarkt', 'Berliner Straße', 'Goetheplatz', 'Goethestraße', 'Zeil', 'Airportring',
                'Platz der Deutschen Einheit', 'Panoramastraße', 'Unter den Linden', 'Dorotheenstraße',
                'Marienstraße', 'Frauenplatz', 'Burgstraße', 'Brunnengasse', 'Kornmarkt', 'Lichthof', 'Opernplatz',
                'Rosmaringasse', 'Salzgasse', 'Schillerstraße', 'Forstweg', 'Karlstraße', 'Domplatz']

CITIES = [('Würzburg', '97074'), ('Frankfurt', '60549'), ('Frankfurt am Main', '60329'), ('Hamburg', '20457'),
          ('Rostock', '18057'), ('Berlin', '10178'), ('München', '80331'), ('Stuttgart', '70173'), ('Nürnberg', '90403'),
          ('Schweinfurt', '97421'), ('Bonn', '53111'), ('Köln', '50667'), ('Hannover', '30159'), ('Dresden', '01067')]

FIRST_NAMES = ['Oliver', 'Max', 'Anna', 'Julia', 'Sabrina', 'Thomas', 'Tom', 'Julian', 'Lukas', 'Peter', 'Marius',
               'Filipp', 'Vanessa', 'Wolfgang', 'Martin', 'Jonas', 'Emil', 'Paul', 'Pauline', 'Alexander', 'Leon', 'Mia',
               'Maria', 'Leonie', 'Johanna', 'Lena', 'Hanna', 'David', 'Elias', 'Daniel', 'Sarah', 'Katharina', 'Kevin',
               'Sebastian', 'Franziska', 'Laura', 'Tobias', 'Patrick', 'Stefanie', 'Christina', 'Nadine', 'Benjamin',
               'Noah', 'Luca', 'Liam', 'Emma', 'Lina', 'Nina', 'Elena', 'Gabriel']

LAST_NAMES = ['Müller', 'Schmidt', 'Mustermann', 'Musterfrau', 'Fischer', 'Schneider', 'Weber', 'Hofmann', 'Meyer',
              'Krause', 'Werner', 'Peters', 'Kaiser', 'Baumann', 'Jung', 'Fuchs', 'Vogt', 'Groß', 'Pohl', 'Busch',
              'Wolff', 'Sauer', 'Pfeffer', 'Kuhn', 'Engel', 'Keller', 'Brown', 'Jones', 'Bauer', 'Egger', 'Ebner',
              'Haas', 'Heilig', 'Lang', 'Binder']


def get_random_address():
    city = random.choice(CITIES)
    return random.choice(STREET_NAMES), random.randint(1, 100), city[0], city[1]


def get_random_name():
    return f'{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}'
