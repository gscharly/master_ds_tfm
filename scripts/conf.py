# Creating dataframe
DATA_PATH = '/home/carlos/MasterDS/tfm/data'

LITERAL_TO_LEAGUE = {
    'premier': 'premier_league',
    'spanish': 'spanish_la_liga',
    'german': 'german_bundesliga',
    'french': 'french_league_one',
    'italian': 'italian_serie_a',
    'champions': 'champions_league',
    'mls': 'mls'
}

MLS_2019_2020 = ['Atlanta United FC', 'Chicago', 'Colorado Rapids', 'Columbus Crew SC', 'D.C. United', 'FC Cincinnati', 'Dallas', 'Houston Dynamo',
                      'Los Angeles Football Club', 'LA Galaxy', 'Minnesota United FC', 'Montreal Impact', 'New England Revolution', 'New York City',
                      'New York Red Bulls', 'Orlando City SC', 'Philadelphia Union', 'Portland Timbers', 'Real Salt Lake', 'San Jose Earthquakes',
                      'Seattle Sounders FC', 'Sporting Kansas City', 'Toronto', 'Vancouver Whitecaps FC', 'USA', 'Jamaica', 'Cuba', 'Martinique', 'Bermuda',
                      'Costa Rica', 'Olimpia de Tegucigalpa', 'Tigres', 'San Carlos', 'Italy', 'Independiente La Chorrera', 'Uruguay', 'Canada', 'Haiti',
                      'Cruz Azul', 'El Salvador', 'Honduras', 'León', 'Venezuela', 'Curaçao', 'Guyana', 'New Mexico United', 'Mexico', 'Panama',
                      'Trinidad', 'Sacramento Republic', 'St. Kitts And Nevis', 'Cavalry', 'Pittsburgh Riverhounds SC', 'Louisville City',
                      'Herediano', 'Nicaragua', 'Monterrey', 'Motagua', 'Austin Bold', 'Saint Louis', 'Santos Laguna', 'MLS Homegrown Team', 'Guadalajara U20',
                      'Nashville', 'Club Tijuana']

MLS_2018_2019 = MLS_2019_2020 + ['US Virgin Islands', 'MLS All-Stars', 'Juventus', 'France', 'Tauro FC', 'Fresno FC', 'Santa Tecla', 'Paraguay', 'Colombia',
                      'Guadalajara', 'San Antonio', 'América', 'Golden State Force', 'Portugal', 'Ottawa Fury',
                                 'Ottawa Fury FC', 'New Zealand', 'NTX Rayados', 'North Carolina', 'England',
                      'Miami United', 'Richmond Kickers', 'Bolivia', 'Brazil', 'Dominica', 'Republic of Ireland', 'Tigres UANL U20', 'Peru', 'Charleston Battery']

MLS_2017_2018 = MLS_2018_2019 + ['Arabe Unido', 'Tulsa Roughnecks', 'Orange County']

MLS_2016_2017 = MLS_2017_2018 + ['Harrisburg City Islanders', 'Fort Lauderdale Strikers', 'Fort Lauderdale', 'Ecuador', 'Alianza', 'Real Estelí', 'New York Cosmos',
                                 'Saprissa', 'Querétaro', 'Argentina', 'Arsenal', 'Antigua GFC', 'Morocco', 'Azerbaijan', 'Mauritania', 'Chile', 'Oklahoma City Energy FC',
                                 'Germany', 'Uzbekistan', 'Rochester Rhinos', 'Sweden', 'Indy Eleven', 'Suchitepéquez', 'La Máquina FC', 'Dragón',
                                 'Carolina RailHawks', 'Guatemala', 'South Korea', 'Wilmington Hammerheads', 'Kitsap Pumas']
MLS_2015_2016 = MLS_2016_2017 + ['Comunicaciones', 'Alajuelense', 'Charlotte Independence', 'Municipal', 'Denmark', 'Club Deportivo Olimpia', 'Netherlands', 'Belize',
                                 'América U20', 'FC Edmonton', 'Montego Bay United']



# Entity names
TEAMS = {
    "premier_league_2019_2020": ['Liverpool', 'Manchester City', 'Leicester City', 'Chelsea', 'Manchester United', 'Wolverhampton Wanderers', 'Sheffield United', 'Tottenham Hotspur',
                                'Arsenal', 'Burnley', 'Crystal Palace', 'Everton', 'Newcastle United', 'Southampton', 'Brighton', 'West Ham', 'Watford', 'Bournemouth',
                                'Aston Villa', 'Norwich City'],
    "premier_league_2018_2019": ['Liverpool', 'Manchester City', 'Leicester City', 'Chelsea', 'Manchester United', 'Wolverhampton Wanderers', 'Fulham', 'Tottenham Hotspur',
                                'Arsenal', 'Burnley', 'Crystal Palace', 'Everton', 'Newcastle United', 'Southampton', 'Brighton', 'West Ham', 'Watford', 'Bournemouth',
                                'Cardiff City', 'Huddersfield Town'],
    "premier_league_2017_2018": ['Liverpool', 'Manchester City', 'Leicester City', 'Chelsea', 'Manchester United', 'Tottenham Hotspur',
                                'Arsenal', 'Burnley', 'Crystal Palace', 'Everton', 'Newcastle United', 'Southampton', 'Brighton', 'West Ham', 'Watford', 'Bournemouth',
                                'Swansea City', 'Huddersfield Town', 'West Bromwich Albion', 'Stoke City'],
    "premier_league_2016_2017": ['Liverpool', 'Manchester City', 'Leicester City', 'Chelsea', 'Manchester United', 'Tottenham Hotspur',
                                'Arsenal', 'Burnley', 'Crystal Palace', 'Everton', 'Sunderland', 'Southampton', 'West Ham', 'Watford', 'Bournemouth',
                                'Swansea City', 'Hull City', 'West Bromwich Albion', 'Stoke City', 'Middlesbrough'],
    "champions_league_2019_2020": ['Barcelona', 'Atlético de Madrid', 'Real Madrid', 'Valencia', 'Manchester City', 'Chelsea', 'Liverpool', 'Tottenham Hotspur', 'Juventus', 'Napoli',
                                   'Atalanta', 'Inter Milan', 'FC Bayern München', 'Borussia Dortmund', 'RB Leipzig', 'Bayer 04 Leverkusen', 'Paris Saint Germain', 'Lille', 'Lyon',
                                   'Zenit St Petersburg', 'Lokomotiv Moscow', 'Benfica', 'Shakhtar Donetsk', 'Galatasaray', 'Red Bull Salzburg', 'Club Brugge', 'Olympiakos',
                                   'Crvena Zvezda', 'Dinamo Zagreb', 'KRC Genk', 'Slavia Prague', 'Ajax'
                                   ],
    "german_bundesliga_2017_2018": ['FC Bayern München', 'Borussia Dortmund', 'RB Leipzig', 'Bayer 04 Leverkusen', 'Werder Bremen', 'Hoffenheim',
                                    'FC Schalke 04', 'Borussia Mönchengladbach', 'Eintracht Frankfurt', 'Hertha Berlin', 'FSV Mainz 05',
                                     'VfL Wolfsburg', 'Sport-Club Freiburg', 'FC Augsburg', 'VfB Stuttgart', 'Hannover', 'FC Köln', 'Hamburger SV'],
    "german_bundesliga_2018_2019": ['FC Bayern München', 'Borussia Dortmund', 'RB Leipzig', 'Bayer 04 Leverkusen', 'Werder Bremen', 'Hoffenheim',
                                    'FC Schalke 04', 'Borussia Mönchengladbach', 'Eintracht Frankfurt', 'Hertha Berlin', 'Fortuna Düsseldorf', '1. FSV Mainz 05',
                                     'VfL Wolfsburg', 'Sport-Club Freiburg', 'FC Augsburg', 'VfB Stuttgart', 'Hannover', 'FC Nürnberg'],
    "german_bundesliga_2019_2020": ['FC Bayern München', 'Borussia Dortmund', 'RB Leipzig', 'Bayer 04 Leverkusen', 'SC Paderborn 07', 'Werder Bremen', 'Hoffenheim',
                                    'FC Schalke 04', 'Borussia Mönchengladbach', 'Eintracht Frankfurt', 'Hertha Berlin', 'Fortuna Düsseldorf', 'FC Köln', 'FSV Mainz 05',
                                     'VfL Wolfsburg', 'Sport-Club Freiburg', 'FC Augsburg', 'FC Union Berlin'],
    "italian_serie_a_2017_2018": ['Atalanta', 'Bologna', 'Cagliari', 'Fiorentina', 'Genoa', 'Inter Milan', 'Juventus', 'Lazio', 'Milan',
                                  'Napoli', 'Roma', 'Sampdoria', 'Sassuolo', 'SPAL', 'Torino', 'Udinese', 'Chievo', 'Crotone', 'Verona', 'Benevento'],
    "italian_serie_a_2018_2019": ['Atalanta', 'Bologna', 'Cagliari', 'Fiorentina', 'Genoa', 'Inter Milan', 'Juventus', 'Lazio', 'Milan',
                                  'Napoli', 'Parma', 'Roma', 'Sampdoria', 'Sassuolo', 'SPAL', 'Torino', 'Udinese', 'Empoli', 'Frosinone', 'Chievo'],
    "italian_serie_a_2019_2020": ['Atalanta', 'Bologna', 'Brescia', 'Cagliari', 'Fiorentina', 'Genoa', 'Verona', 'Inter Milan', 'Juventus', 'Lazio', 'Lecce', 'Milan',
                                  'Napoli', 'Parma', 'Roma', 'Sampdoria', 'Sassuolo', 'SPAL', 'Torino', 'Udinese'],
    "mls_2015_2016": MLS_2015_2016,
    "mls_2016_2017": MLS_2016_2017,
    "mls_2017_2018": MLS_2017_2018,
    "mls_2018_2019": MLS_2018_2019,
    "mls_2019_2020": MLS_2019_2020,
    "spanish_la_liga_2017_2018": ['Athletic Club', 'Atlético de Madrid', 'Leganés', 'Alavés', 'Barcelona', 'Getafe', 'Levante',
                                  'Celta de Vigo', 'Espanyol', 'Real Betis', 'Real Madrid', 'Real Sociedad', 'Eibar', 'Sevilla', 'Valencia', 'Villarreal',
                                  'Girona', 'Deportivo de La Coruña', 'Las Palmas', 'Málaga'],
    "spanish_la_liga_2018_2019": ['Athletic Club', 'Atlético de Madrid', 'Leganés', 'Alavés', 'Barcelona', 'Getafe', 'Levante',
                                  'Celta de Vigo', 'Espanyol', 'Real Betis', 'Real Madrid', 'Real Sociedad', 'Valladolid', 'Eibar', 'Sevilla', 'Valencia', 'Villarreal',
                                  'Girona', 'Huesca', 'Rayo Vallecano'],
    "spanish_la_liga_2019_2020": ['Athletic Club', 'Atlético de Madrid', 'Leganés', 'Osasuna', 'Alavés', 'Barcelona', 'Getafe', 'Granada CF', 'Levante', 'Mallorca',
                                  'Celta de Vigo', 'Espanyol', 'Real Betis', 'Real Madrid', 'Real Sociedad', 'Valladolid', 'Eibar', 'Sevilla', 'Valencia', 'Villarreal'],
    "french_ligue_one_2019_2020": ['Monaco', 'St Etienne', 'Amiens', 'Angers', 'Dijon', 'Bordeaux', 'Metz', 'Nantes', 'Lille', 'Montpellier', 'Nîmes', 'Nice',
                                   'Lyon', 'Marseille', 'Paris Saint Germain', 'Strasbourg', 'Brest', 'Reims', 'Rennes', 'Toulouse'],

}

EN_LABELS = {
    'PLAYER': ['PERSON'],
    'TEAM': ['ORG', 'GPE', 'PERSON', 'CARDINAL', 'NORP', 'EVENT', 'FAC', 'LOC']
}
CSV_DATA_PATH = '{}/csv'.format(DATA_PATH)

SUMMARY_PATH = CSV_DATA_PATH + '/summaries'
ARTICLES_PATH = CSV_DATA_PATH + '/articles_events.csv'
METRICS_PATH = DATA_PATH + '/metrics'
