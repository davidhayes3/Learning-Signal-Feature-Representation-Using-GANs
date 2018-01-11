\ProvidesClass{monthly-report}
\LoadClass[12pt, fleqn]{article}
\usepackage{hyperref, soul, color}
\DeclareRobustCommand{\hlcyan}[1]{{\sethlcolor{cyan}\hl{#1}}}
\usepackage{graphicx,amssymb}
\usepackage[nodisplayskipstretch]{setspace}
\usepackage{cite}
\usepackage[paperwidth=21cm,paperheight=29.7cm,includehead,headheight=2.2cm,pdftex,hmargin={2cm,2cm},vmargin={0cm,2cm},]{geometry} 
\parskip = 0.5em
\parindent = 0.0ex
\renewcommand\familydefault{\sfdefault}
\def\bil{\begin{flushleft}\parsep=0em\topsep=0em\begin{itemize}\itemsep=0em}
\def\eil{\end{itemize}\end{flushleft}}
\def\bi{\parsep=0em\topsep=0em\begin{itemize}\itemsep=0em}
\def\ei{\end{itemize}}
\def\bin{\parsep=0em\topsep=0em\begin{enumerate}\itemsep=0em}
\def\ein{\end{enumerate}}
\def\bb{\item}
\def\bd{\item[{\small\bf--}]}
\renewcommand{\labelitemi}{\raisebox{0.5mm}{\scriptsize$\square$}}
\renewcommand{\labelitemii}{\raisebox{0.5mm}{\small\bf--}}
\def\BibTeX{{ B\kern-.05em{i\kern-.025em b}\kern-.08em T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

\renewcommand{\labelitemiii}{\raisebox{0.5mm}{\color{darkorange}\small$\bullet$}}
\renewcommand{\labelitemiv}{\raisebox{0.5mm}{\color{darkorange}\small$\circ$}}
\usepackage{fancyheadings}
\renewcommand{\headrulewidth}{0.5pt}
\renewcommand{\contentsname}{\normalsize\bf Table of Contents}
\chead{}\rhead{\small \thepage/\pageref{last_page}}\lhead{\small Monthly Report \monthlyreportnumber\ (\studentname~--~\ucdstudentnumber)}\rfoot{}\lfoot{}\cfoot{} 
\pagestyle{fancy} 

\setcounter{tocdepth}{2}

\makeatletter
\renewcommand{\l@section}{\@dottedtocline{1}{0em}{2.3em}}
\makeatother


\def\maketitle{\begin{titlepage}
\thispagestyle{empty}
\let\footnotesize\small \let\footnoterule\relax \setcounter{page}{0}
\null
\vskip -2cm
\begin{minipage}{0.15\textwidth}
\includegraphics[height=3cm]{logo}
\end{minipage}
\begin{minipage}{0.8\textwidth}
\setstretch{1.3} \raggedright
{\Large\bf \modulename: }{\Large\projecttitle}\\[0.3cm]
{\Large{\bf\studentname}} {\Large (UCD: \ucdstudentnumber)}\\
{\Large Monthly Report \monthlyreportnumber, \today }
\vfil
\end{minipage}
%\renewcommand\baselinestretch{1}

\vspace*{0.7cm}
\rule{\textwidth}{0.5pt}

{\tableofcontents }
\vfil
\null
\end{titlepage}}