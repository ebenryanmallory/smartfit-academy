export interface TestPrepLessonPlan {
  title: string;
  description: string;
  lessons: string[];
}

export const gedLessonPlans: TestPrepLessonPlan[] = [
  {
    title: 'Whole-Number Foundations & Number Sense',
    description: 'Core fluency with place value, operations, factors and multiples—bedrock for all GED math.',
    lessons: [
      'Place Value, Rounding & Estimation',
      'Four-Operation Arithmetic Mastery',
      'Prime Factorization & Divisibility Rules',
      'Greatest Common Factor vs. Least Common Multiple',
      'Order of Operations in Multi-Step Problems'
    ]
  },
  {
    title: 'Fractions, Decimals & Percents',
    description: 'Convert, compare and compute with rational numbers in real-life contexts.',
    lessons: [
      'Fraction Concepts & Visual Models',
      'Adding, Subtracting, Multiplying & Dividing Fractions',
      'Decimal Operations & Scientific Notation',
      'Percent–Fraction–Decimal Conversions',
      'Percent Increase, Decrease & Error'
    ]
  },
  {
    title: 'Ratio, Proportion & Quantitative Problem Solving',
    description: 'Solve proportional-reasoning problems, scale drawings and rate questions.',
    lessons: [
      'Ratios & Rates in Everyday Contexts',
      'Solving Proportions Algebraically',
      'Unit Rates, Speed & Density',
      'Scale Factors & Similar Figures',
      'Dimensional Analysis for Unit Conversion'
    ]
  },
  {
    title: 'Introductory Algebra',
    description: 'Translate real-world stories into algebraic expressions and one-variable equations.',
    lessons: [
      'Variables, Expressions & Translating Words to Math',
      'Solving One-Step & Multi-Step Linear Equations',
      'Linear Inequalities & Solution Sets',
      'Graphing Lines on the Coordinate Plane',
      'Slope, Rate of Change & Intercepts'
    ]
  },
  {
    title: 'Intermediate Algebra & Functions',
    description: 'Manipulate polynomials and model with linear & quadratic functions.',
    lessons: [
      'Polynomials & Exponent Rules',
      'Factoring Techniques & Zero Product Property',
      'Systems of Equations—Substitution & Elimination',
      'Quadratic Functions & Parabolas',
      'Function Notation & Tables'
    ]
  },
  {
    title: 'Geometry & Measurement',
    description: 'Compute perimeter, area, volume and use the Pythagorean theorem in context.',
    lessons: [
      '2-D Figures: Triangles, Quadrilaterals & Circles',
      'Area & Circumference Applications',
      'Surface Area & Volume of Prisms & Cylinders',
      'Pythagorean Theorem & Distance Formula',
      'Coordinate Geometry & Transformations'
    ]
  },
  {
    title: 'Data Analysis, Statistics & Probability',
    description: 'Interpret charts and summarize data; evaluate chance events.',
    lessons: [
      'Statistical Displays: Bar, Line & Circle Graphs',
      'Measures of Center & Spread (IQR & MAD)',
      'Box-and-Whisker & Histogram Interpretation',
      'Basic Probability & Compound Events',
      'Making Data-Driven Predictions'
    ]
  },
  {
    title: 'Science Literacy & Reasoning Skills',
    description: 'Scientific method, hypothesis testing and interpreting experimental data.',
    lessons: [
      'Variables & Experimental Design',
      'Reading Tables, Graphs & Scientific Diagrams',
      'Cause-and-Effect vs. Correlation',
      'Energy, Matter & Basic Physical Principles',
      'Life & Earth-Science Data Sets'
    ]
  },
  {
    title: 'Social Studies Analysis Skills',
    description: 'Use evidence from history, civics, economics & geography passages.',
    lessons: [
      'Reading Historical Documents & Primary Sources',
      'Civics: U.S. Government Structure & Rights',
      'Economics: Supply, Demand & Market Structures',
      'Geography: Maps, Charts & Spatial Thinking',
      'Evaluating Bias & Point of View'
    ]
  },
  {
    title: 'Reading Comprehension for Evidence-Based Responses',
    description: 'Identify central ideas, analyze arguments and cite textual evidence.',
    lessons: [
      'Main Idea vs. Supporting Detail',
      'Author\'s Purpose & Tone',
      'Inference, Implicit Meaning & Reasoning',
      'Comparing Texts on a Similar Topic',
      'Critical Reading of Informational Graphics'
    ]
  },
  {
    title: 'Grammar, Usage & Mechanics',
    description: 'Edit and revise sentences for clarity, correctness and style.',
    lessons: [
      'Sentence Boundaries & Fragments',
      'Subject–Verb & Pronoun-Antecedent Agreement',
      'Punctuation: Commas, Semicolons & Apostrophes',
      'Modifiers & Parallel Structure',
      'Word Choice, Formal vs. Informal Tone'
    ]
  },
  {
    title: 'Extended Response (GED Essay) Mastery',
    description: 'Plan and compose a coherent, evidence-based argumentative essay.',
    lessons: [
      'Understanding the GED Essay Prompt & Rubric',
      'Planning: Brainstorming & Outlining Quickly',
      'Writing a Focused Thesis & Topic Sentences',
      'Integrating & Citing Source Evidence',
      'Revision Strategies Under Time Pressure'
    ]
  },
  {
    title: 'Advanced Geometry & Coordinate Systems',
    description: 'Deepen spatial reasoning with composite 2-D/3-D figures, coordinate proofs and intro-trig.',
    lessons: [
      'Similarity & Congruence Proofs',
      'Sector & Segment Problems in Circles',
      'Cross-Sections, Nets & 3-D Modeling',
      'Coordinate Proofs & Transformations',
      'Right-Triangle Trigonometry Basics',
      'Real-World Geometry Modeling'
    ]
  },
  {
    title: 'Probability, Combinatorics & Statistical Reasoning',
    description: 'Multi-step probability, counting principles and interpreting real statistical studies.',
    lessons: [
      'Sample Spaces, Events & Venn Diagrams',
      'Permutations, Combinations & Counting Principle',
      'Binomial & Geometric Probability Models',
      'Confidence Intervals & Margin of Error',
      'Correlation vs. Causation in Studies',
      'Linear Regression & Residual Analysis'
    ]
  },
  {
    title: 'Workforce & Financial Literacy',
    description: 'Mathematics of paychecks, budgeting, credit and workplace data.',
    lessons: [
      'Gross vs. Net Pay & Payroll Deductions',
      'Budget Creation & Cash-Flow Analysis',
      'Simple vs. Compound Interest',
      'Loans, Credit Scores & Amortization',
      'Reading Technical Graphs & Tables',
      'Interpreting Workplace Charts & Manuals'
    ]
  },
  {
    title: 'Digital & Information Literacy',
    description: 'Locate, evaluate and cite online information accurately and ethically.',
    lessons: [
      'Source Credibility & Lateral Reading',
      'Infographics & Interactive Media Analysis',
      'Advanced Web-Search Operators',
      'Summarising, Paraphrasing & Quoting',
      'Plagiarism Avoidance & Citation Basics'
    ]
  },
  {
    title: 'Reading Passage Types & Mapping',
    description: 'Fine-tune strategy for Literature, History, Social Science and Science passages.',
    lessons: [
      'Literary Narrative Structure & Tone',
      'Historical Documents — Paired Passages',
      'Social-Science Argument Analysis',
      'Science Passage Data Integration',
      'Passage Mapping & Annotation Drills',
      'Dual-Passage Synthesis Questions'
    ]
  },
  {
    title: 'Rhetorical Synthesis & Quantitative Evidence',
    description: 'Integrate charts/graphs with text and evaluate an author\'s argumentative moves.',
    lessons: [
      'Text-Chart Alignment Questions',
      'Purpose, Strategy & Audience Analysis',
      'Strengthening / Weakening an Argument',
      'Logical Fallacies & Counterclaims',
      'Selecting Concluding Statements',
      'Precision & Formal Style in Revisions'
    ]
  }
];

export const satLessonPlans: TestPrepLessonPlan[] = [
  {
    title: 'Heart of Algebra',
    description: 'Linear equations, inequalities and systems—the backbone of SAT Math.',
    lessons: [
      'Linear Expressions & Rearranging Formulas',
      'Solving Single-Variable Linear Equations',
      'Graphing Lines & Understanding Slope',
      'Systems of Linear Equations & Inequalities',
      'Word Problems: Rate, Work & Mixture'
    ]
  },
  {
    title: 'Problem Solving & Data Analysis',
    description: 'Ratios, percentages and data interpretation with an emphasis on real-world contexts.',
    lessons: [
      'Ratios, Rates & Proportional Relationships',
      'Percent Change & Compound Percent',
      'Tables, Scatterplots & Two-Way Tables',
      'Statistical Measures & Data Inference',
      'Probability & Expected Value'
    ]
  },
  {
    title: 'Passport to Advanced Math',
    description: 'Quadratics, exponentials and algebraic structure needed for STEM majors.',
    lessons: [
      'Functions & Their Notation',
      'Quadratic Equations: Factoring & Quadratic Formula',
      'Exponential Growth & Decay Models',
      'Radical & Rational Expressions',
      'Polynomial Division & Remainder Theorem'
    ]
  },
  {
    title: 'Additional Topics in Math',
    description: 'Geometry, trigonometry and complex numbers that round out SAT content.',
    lessons: [
      'Circle Theorems & Arc/Chord Relationships',
      'Area & Volume of 2-D/3-D Figures',
      'Right Triangle Trigonometry & SOH-CAH-TOA',
      'Radian Measure & Unit Circle Basics',
      'Complex Numbers & Operations'
    ]
  },
  {
    title: 'Command of Evidence',
    description: 'Locate and use textual evidence to support answers and arguments.',
    lessons: [
      'Line-Reference & Evidence Questions',
      'Data Graphics Paired with Passages',
      'Evaluating an Author\'s Claim',
      'Supporting Quantitative Evidence',
      'Eliminating Distractors Efficiently'
    ]
  },
  {
    title: 'Words in Context & Vocabulary',
    description: 'Leverage context clues to choose precise academic vocabulary.',
    lessons: [
      'Contextual Meaning vs. Common Meaning',
      'Connotation & Register',
      'Strategy for "Most Nearly Means" Questions',
      'Transition Words & Structural Signals',
      'Building Academic Word Families'
    ]
  },
  {
    title: 'Expression of Ideas',
    description: 'Revise texts for logic, cohesion and rhetorical effectiveness.',
    lessons: [
      'Organizing Paragraphs & Logical Flow',
      'Adding, Revising or Deleting Sentences',
      'Effective Introductions & Conclusions',
      'Precision and Conciseness in Language',
      'Supporting Evidence with Data & Charts'
    ]
  },
  {
    title: 'Standard English Conventions',
    description: 'Grammar and mechanics tested in the Writing & Language section.',
    lessons: [
      'Sentence Boundaries & Run-ons',
      'Verb Tense, Mood & Voice',
      'Pronouns & Agreement',
      'Modifiers & Parallelism',
      'Punctuation for Lists & Clauses'
    ]
  },
  {
    title: 'SAT Analytical Essay (Optional)',
    description: 'Craft a rhetorical analysis of an argumentative passage.',
    lessons: [
      'Understanding the Essay Task & Scoring',
      'Annotating the Passage for Rhetorical Devices',
      'Structuring the Introduction & Thesis',
      'Developing Body Paragraphs with Evidence',
      'Polishing Language under Time Constraints'
    ]
  },
  {
    title: 'Test Strategy, Timing & Pacing',
    description: 'Optimize section order, guessing strategy and mental stamina.',
    lessons: [
      'Setting Section-by-Section Time Benchmarks',
      'Grid-In and Multiple-Choice Best Practices',
      'Calculator vs. No-Calculator Section Tips',
      'Bubble-Sheet & Answer-Changing Strategy',
      'Mindset, Stress-Management & Sleep Hygiene'
    ]
  },
  {
    title: 'Statistics, Probability & Data Modeling',
    description: 'All statistics and probability the SAT may test—now in one dedicated plan.',
    lessons: [
      'Center, Spread & Shape of Distributions',
      'Sampling Methods & Bias Detection',
      'Two-Way Tables & Conditional Probability',
      'Least-Squares Regression & Residuals',
      'Expected Value & Decision Making',
      'Designing Experiments & Surveys'
    ]
  },
  {
    title: 'Geometry & Trigonometry Deep Dive',
    description: 'Comprehensive review of plane and solid geometry plus trigonometric relationships required for SAT success.',
    lessons: [
      'Geometric Theorems & Proofs',
      'Circle Properties & Arc Length',
      'Volume & Surface Area of Solids',
      'Trigonometric Ratios & Unit Circle',
      'Modeling with Trigonometric Functions',
      'Geometric Transformations & Symmetry'
    ]
  },
  {
    title: 'Reading Passage Types & Annotation Strategies',
    description: 'Tailor approach for Literature, History/Social Science and Science passages through effective mapping and annotation.',
    lessons: [
      'Literary Narrative Structure & Themes',
      'Historical Documents Analysis',
      'Social Science Argument Evaluation',
      'Science Passage Data Questions',
      'Passage Mapping Techniques',
      'Synthesizing Information Across Passages'
    ]
  },
  {
    title: 'Evidence Synthesis from Graphics & Text',
    description: 'Integrate textual and graphical information to answer data-driven questions accurately.',
    lessons: [
      'Interpreting Charts & Tables within Passages',
      'Aligning Quantitative Data with Claims',
      'Spotting Misleading Graphs & Figures',
      'Selecting Supporting Lines & Data Points',
      'Strengthening or Weakening Arguments',
      'Formulating Conclusions from Mixed Media'
    ]
  },
  {
    title: 'Digital Literacy & Information Evaluation',
    description: 'Develop critical skills for finding, assessing and citing reliable online information.',
    lessons: [
      'Source Credibility & Lateral Reading',
      'Advanced Web-Search Operators',
      'Fact-Checking and Verification',
      'Paraphrasing & Quoting Accurately',
      'Avoiding Plagiarism',
      'MLA & APA Citation Basics'
    ]
  },
  {
    title: 'Mindset & Test-Day Readiness',
    description: 'Build the psychological resilience and logistical plan required to perform at your best on test day.',
    lessons: [
      'Growth Mindset & Motivation',
      'Stress-Reduction & Focus Techniques',
      'Sleep, Nutrition & Exercise Guidelines',
      'Creating a Personal Timing Strategy',
      'Strategic Guessing & Answer-Changing',
      'Post-Test Reflection & Next Steps'
    ]
  }
];

export const actLessonPlans: TestPrepLessonPlan[] = [
  {
    title: 'ACT English: Usage & Mechanics',
    description: 'Grammar, punctuation and sentence structure for the 75-question English test.',
    lessons: [
      'Comma, Semicolon & Colon Rules',
      'Verb Forms & Subject–Verb Agreement',
      'Pronoun Clarity & Agreement',
      'Modifiers & Parallel Construction',
      'Diction, Idioms & Register'
    ]
  },
  {
    title: 'ACT English: Rhetorical Skills',
    description: 'Strategy, organization and style questions that test effective writing.',
    lessons: [
      'Topic & Transitional Sentences',
      'Logical Paragraph Order',
      'Add/Delete/Revise Decisions',
      'Conciseness & Redundancy',
      'Tone, Style & Audience Awareness'
    ]
  },
  {
    title: 'Pre-Algebra & Elementary Algebra',
    description: 'The first 20–25 ACT math questions focus on foundational algebra skills.',
    lessons: [
      'Prime Factorization & Number Properties',
      'Ratio, Rate & Proportion Problems',
      'Linear Equations & Inequalities',
      'Exponents, Roots & Scientific Notation',
      'Basic Word Problems & Translating Phrases'
    ]
  },
  {
    title: 'Intermediate Algebra & Coordinate Geometry',
    description: 'Functions, quadratics and graph interpretation—middle difficulty questions.',
    lessons: [
      'Quadratic Functions & Parabolas',
      'Polynomial Factoring & Zeros',
      'Systems of Equations & Inequalities',
      'Graphing Rational & Radical Functions',
      'Slope, Distance & Midpoint Formula'
    ]
  },
  {
    title: 'Plane Geometry & Trigonometry',
    description: 'Last third of ACT math—geometric reasoning and basic trig.',
    lessons: [
      'Angles, Triangles & Circle Theorems',
      'Area & Volume of Common Figures',
      'Coordinate Plane Geometry Applications',
      'Basic Trigonometric Ratios & Identities',
      'Law of Sines/Cosines Word Problems'
    ]
  },
  {
    title: 'ACT Reading: Passage Strategies',
    description: 'Approach each of the four passage types with a tailored method.',
    lessons: [
      'Prose Fiction / Literary Narrative Techniques',
      'Social Science Passage Mapping',
      'Humanities Passage Tone & Perspective',
      'Natural Science Passage Data Integration',
      'Timing & Question Prioritization'
    ]
  },
  {
    title: 'ACT Science: Data Representation',
    description: 'Interpret charts, tables and graphs quickly and accurately.',
    lessons: [
      'Graph & Table Decoding Skills',
      'Identifying Trends & Outliers',
      'Interpolating & Extrapolating Data',
      'Mathematical Relationships in Graphs',
      'Common Trap Answers & How to Avoid Them'
    ]
  },
  {
    title: 'ACT Science: Research Summaries',
    description: 'Understand experimental setups and compare results.',
    lessons: [
      'Dissecting Experimental Design',
      'Independent vs. Dependent Variables',
      'Control Groups & Reproducibility',
      'Multiple-Experiment Comparison',
      'Synthesizing Conclusions Quickly'
    ]
  },
  {
    title: 'ACT Science: Conflicting Viewpoints',
    description: 'Analyze passages presenting opposing scientific hypotheses.',
    lessons: [
      'Reading for Hypotheses & Claims',
      'Mapping Agreements & Disagreements',
      'Evaluating Evidence for Each View',
      'Recognizing Underlying Assumptions',
      'Selecting Impartial Conclusions'
    ]
  },
  {
    title: 'ACT Writing (Optional Essay)',
    description: 'Plan and compose a well-structured argumentative essay in 40 minutes.',
    lessons: [
      'Understanding the Prompt & Perspectives',
      'Brainstorming & Organizing Quickly',
      'Crafting a Clear Thesis & Roadmap',
      'Developing Body Paragraphs with Evidence',
      'Editing for Language & Time Management'
    ]
  },
  {
    title: 'ACT Test Management & Strategy',
    description: 'Pacing plans, guessing strategy and section order decisions.',
    lessons: [
      'Section Timing Benchmarks',
      'Answer-Sheet Bubbling Efficiency',
      'Intelligent Guessing & Letter of the Day',
      'Maintaining Focus Across 4 Straight Sections',
      'Pre-Test Routine & Anxiety Reduction'
    ]
  },
  {
    title: 'Statistics & Probability (ACT Math/Science)',
    description: 'Master statistics questions that appear in both Math and Science sections.',
    lessons: [
      'Data Displays & Descriptive Statistics',
      'Correlation, Outliers & Line of Best Fit',
      'Probability Rules & Counting Principle',
      'Combinations, Permutations & Expected Value',
      'Experimental vs. Theoretical Probability',
      'Margin of Error & Study Reliability'
    ]
  },
  {
    title: 'Vectors, Matrices & Advanced Topics',
    description: 'Higher-difficulty content that shows up in the final 10 ACT Math questions.',
    lessons: [
      'Vector Addition, Components & Applications',
      'Dot Product & Magnitude',
      '2×2 and 3×3 Matrix Operations',
      'Determinant & Inverse Matrices',
      'Transformations Using Matrices',
      'Parametric & Polar Quick Review'
    ]
  },
  {
    title: 'ACT Science: Experimental Design Mastery',
    description: 'Dissect and evaluate complex research setups in the Science section.',
    lessons: [
      'Hypothesis Generation & Variable Control',
      'Determining Independent & Dependent Variables',
      'Identifying Experimental Flaws',
      'Comparing Multiple Experiments',
      'Drawing Valid Conclusions from Data',
      'Designing Follow-Up Experiments'
    ]
  }
];

// Main function to get test prep lesson plans by topic
export const getTestPrepLessonPlans = (topic: string): TestPrepLessonPlan[] => {
  const topicLower = topic.toLowerCase();
  
  if (topicLower.includes('ged')) {
    return gedLessonPlans;
  } else if (topicLower.includes('sat')) {
    return satLessonPlans;
  } else if (topicLower.includes('act')) {
    return actLessonPlans;
  }
  
  return [];
}; 