# Allocation_de_portefeuille_sous_contrainte_ESG

## Présentation du projet

Les critères Environnementaux, Sociaux et de Gouvernance (ESG) jouent un rôle crucial dans l’évaluation de la performance et de l’impact des entreprises sur la société. L’industrie de la gestion d’actifs s’est dotée de nombreux labels pour répondre aux attentes des investisseurs en matière d’ESG et d’investissement vert.
L’allocation de portefeuille consiste à chercher la meilleur répartition de richesse d’un investisseur parmi un paniers d’actifs, en fonction de ses objectifs et son appétit au risque. L’approche mean-variance de Markowitz, constitue le cadre de référence en matière d’allocation de portefeuille. Elle offre une stratégie de répartition de richesse pour laquelle le rendement attendu est maximal pour un niveau de risque donné. Cependant le simple consensus entre rendement et risque peut s’avérer insuffisant pour pleinement satisfaire les besoins des investisseurs. En effet, ces derniers sont de plus en plus regardants sur l’impact social et environnemental de leurs placements et peuvent même privilégier ces critères à la recherche de performance. L’objectif de ce projet est d’intégrer des scores ESG ainsi que des indices de confiances portant sur ces scores dans le processus d’allocation. Il sera demandé au étudiants de définir et de comparer des méthodes permettant d’intégrer et de combiner au mieux l’aversion au risque et la sensibilité aux critères ESG de l’investisseur.


## Données

Afin de proposer une méthode d’allocation, les étudiants utiliseront la base https://fr.finance.yahoo.com/. Cette dernière fournit les cours des actifs, leur score ESG et de controverse.

Un scraper dynamique des scores ESG et de controverse a été codé à partir de : https://www.msci.com/our-solutions/esg-investing/esg-ratings-climate-search-tool.


## Étapes du projet

- Se familiariser avec les score ESG et de controverse
- Estimer les inputs nécessaires aux méthodes d’allocations
- Comparer plusieurs méthodes d’allocation de portefeuilles et d’intégration des critères ESG
- Étudier la modification des frontières efficientes en fonction de ces contraintes

# Comment naviguer dans ce repo ?

Les codes pertinents concernent:
- Le scraper MSCI pour les scores ESG et les scores de controverse des entreprises [exploration/scraper]
- La class Portfolio créée pour toutes notes analyses [src/portfolio_class.py]
- Le notebook d'étude des scores ESG avec des frontières efficientes, sharpe ratio et composition de portefeuille efficients (d'où proviennent les représentations du rapport) [exploration/notebook_allocation_optimale.ipynb]
- Le notebook d'étude des scores de controverse avec des simulations [exploration/notebook_integration_de_la_controverse.ipynb]
