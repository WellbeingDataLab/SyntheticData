# Wellbeing DataLab

Wellbeing DataLab tarjoaa käyttäjilleen alustan, jolla voi kokeilla kuinka hyvin synteettinen data
soveltuu omaan alaan, matalalla kynnyksellä.
Alusta koostuu kahdesta osa-alueesta, synteettisen datan generoinnista, ja sen laadun tarkistamisesta
eli validoinnista.
Näiden ohjeistukset ja koodit ovat omissa kansioissaan, ja tässä dokumentissa on yleisinformaatiota
synteettisestä datasta.

## Mitä on synteettinen data?

- "*Synthetic data is data that has been generated using a purposebuilt mathematical model or algorithm, with the aim of solving a (set of) data science task(s)*", Jordon et al. (2022)

Määritelmän mukaan synteettinen data on siis dataa joka on on generoitu tarkoituksenmukaisesti ratkomaan datatieteiden ongelmia sitä varten rakennetulla mallilla.
Se ei rajaa pois erilaisia datatyyppejä, ja synteettinen data voi siis olla taulukkomuotoisen datan lisäksi
tekstiä, kuvaa, videota, ääntä, signaalia jne., tai vaikka näiden sekoitusta.
Tarkoituksenmukaisuus datatieteiden ongelmanratkontaan tarkoittaa kuitenkin että kaikki koneellisesti generoitu data ei ole synteettistä dataa tässä
merkityksessä, ja esimerkiksi keskustelut Chat-GPT:n kanssa voidaan jättää määritelmän ulkopuolelle.
Toisaalta tietokonesimulaatioita pidetään omana tutkimusalanaan, vaikka niiden generoima data
määritelmän täyttääkin.

Mitä synteettinen data on siis käytännössä? Se voidaan määritellä myös aidon datan jäljitelmäksi, ja synteettistä dataa
tutkimalla saavutaan samoihin johtopäätöksiin kuin aidolla datalla.
Synteettinen data on siis aidon kopio, joka noudattaa mahdollisimman tarkasti sen ominaisuuksia kuten jakaumia ja korrelaatioita,
ja jota voidaan käyttää ratkaisemaan samoja ongelmia kuin aitoa data.
Etuna tällä on luonnollisesti se että aitoa dataa ei tarvitse käyttää, joko tietosuojasyistä tai jos niitä
ei katsota riittävän edustaviksi.
Heikkoutena synteettisellä datalla on sen tarkkuus: Siinä missä jokin aito datanäyte on aina approksimaatio
koko populaatiosta tai jostain ilmiöstä, on synteettinen data approksimaatio tästä aidosta datanäytteestä.

Synteettinen data tarvitsee aina aitoa dataa jonka pohjalta se generoidaan.
Se **ei** siis ratkaise datan puutetta.
Sillä ei myöskään voi lähtökohtaisesti lisätä näytteiden määrää, eli kymmenen henkilön pohjalta generoidut
miljoona synteettistä henkilöä eivät yhdessä edusta pienen valtion populaatiota.
Lisäksi synteettisen datan pohjalta tehtyjen johtopäätöksien kanssa täytyy olla varovainen, sillä
yleistyminen reaalimaailmaan on aina riippuvainen aidosta datasta ja kuinka hyvin sen jäljittelemisessä
on onnistuttu.
Käyttötarkoitusten valinnassa on siis oltava varovainen.

## Mihin sitä voi käyttää?

EU:n yleisen tietosuoja-asetuksen mukaan synteettistä dataa saa käyttää silloin kun käyttäjällä on 
oikeudet alkuperäiseen dataan.
Tämä rajoittaa huomattavasti synteettisen datan käyttötarkoituksia, ja lisää tutkimusta etenkin
turvallisen synteettisen datan generointiin tarvitaan.
Synteettistä dataa voidaan kuitenkin käyttää myös datan augmentoitiin,
eli näytteiden, joita ei esiinny alkuperäisessä datassa, lisäämiseen dataan.
Tämä käyttötapa ei vaadi takeita yksityisyyden säilyttämisestä ja
on sovellettavissa myös tällä hetkellä.

Datan augmentointia käytetään yleisesti koneoppimisessa mallien suorituskyvyn parantamiseen.
Esimerkiksi kuvantunnistuksessa opetusaineistoon voidaan lisätä muunneltuja kuvia, kuten kierrettyjä, 
peilattuja, ylösalaisin olevia, tai vaikka mustavalkoisia kuvia.
Mustavalkoinen ja ylösalaisin oleva kuva koirasta esittää siinä missä muuntelematon, ja
malli voi oppia helpommin sietämään poikkeuksellisia kuvia vaikka niitä ei alkuperäisessä aineistossa
esiintyisikään.

Augmentointi synteettisellä datalla on noudattaa samaa perusperiaatetta. Se vaatii kuitenkin
suurempaa huolellisuutta, sillä nyt uudet näytteet eivät ole suoraan säännönmukaisesti
generoituja, vaan ovat riippuvaisia generoivasta mallista.
Heikkolaatuiset synteettiset näytteet voivat vääristää esimerkiksi ennustavaa mallia jos generoivan mallin
oppima jakauma ei noudatakaan aidon datan taustalla olevaa populaation jakaumaa.
Generoivat mallit saattavat myös generoida mahdottomia tapauksia, mikäli generoinnissa ei oteta näitä
huomioon esimerkiksi poistamalla negatiiviset arvot terveysdatasta.

Käytännössä augmentointia voidaan soveltaa esimerkiksi datasetin tasapainottamiseen.
Kuvitellaan että datasetissämme on 1:9 suhteella miehiä ja naisia.
Nyt voidaan generoida aliedustetusta ryhmästä lisää näytteitä jolloin malli toivon mukaan
ennustaa tasa-arvoisemmin.
Samaa voidaan soveltaa myös esimerkkitapausten generointiin.
Mikäli datasta ei löydy jotain hyvää näytettä esimerkiksi opetusmateriaaliksi,
voidaan sellainen generoida.

Intuitiivisempi käyttötapa on kuitenkin yksityisyyden suojaaminen.
Tällöin synteettistä dataa käytetään aidon datan sijaan, ja teoriassa yksityisyydensuoja säilyy.
Huomattavin etu anonymisointiin on se, että datasta ei jouduta poistamaan muuttujia tai
muokkaamaan informaatiota, vaan data on samalla tavalla käytettävissä kuin aito data.
Heikkoutena on se että on jälleen haastavaa olla varma datan laadusta, jonka lisäksi
täytyy vielä varmistua yksityisyyden säilymisestä.
Synteettinen data on luotu aidon datan perusteella, joten se aina sisältää jotain
informaatiota alkuperäisestä.
Täytyy siis olla jollain osin varma kuinka sensitiivistä informaatiota
on vuotanut synteettiseen dataan.


