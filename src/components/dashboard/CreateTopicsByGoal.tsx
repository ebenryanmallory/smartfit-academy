import GoalButton from './GoalButton';

interface CreateTopicsByGoalProps {
  onGoalSelect: (topic: string, metaTopic: string, subject?: string) => void;
}

const goalData = [
  {
    id: 'SAT-Math',
    title: 'SAT Prep - Math',
    topic: 'SAT Test Preparation',
    metaTopic: 'SAT',
    subject: 'math',
    description: 'SAT preparation is done along with test prep courses via Khan Academy',
    links: [
      {
        text: 'Khan Academy - SAT Math',
        url: 'https://www.khanacademy.org/test-prep/v2-sat-math'
      }
    ]
  },
  {
    id: 'SAT-Reading',
    title: 'SAT Prep - Reading & Writing',
    topic: 'SAT Test Preparation',
    metaTopic: 'SAT',
    subject: 'reading',
    description: 'SAT preparation is done along with test prep courses via Khan Academy',
    links: [
      {
        text: 'Khan Academy - SAT Reading and Writing',
        url: 'https://www.khanacademy.org/test-prep/sat-reading-and-writing'
      }
    ]
  },
  {
    id: 'GED',
    title: 'GED Prep',
    topic: 'GED Test Preparation',
    metaTopic: 'GED',
    description: 'GED preparation resources and practice tests',
    links: [
      {
        text: 'Official GED Website',
        url: 'https://www.ged.com/'
      },
      {
        text: 'GED Practice Test',
        url: 'https://www.ged.com/study/free_online_ged_test/'
      }
    ]
  },
  {
    id: 'ACT',
    title: 'ACT Prep',
    topic: 'ACT Test Preparation',
    metaTopic: 'ACT',
    description: 'ACT preparation resources and practice tests',
    links: [
      {
        text: 'Official ACT Website',
        url: 'https://www.act.org/'
      },
      {
        text: 'Sample ACT Questions',
        url: 'https://www.act.org/content/act/en/products-and-services/the-act/test-preparation/free-act-test-prep/act-online-test-sample-questions.html'
      }
    ]
  }
];

export default function CreateTopicsByGoal({ onGoalSelect }: CreateTopicsByGoalProps) {
  return (
    <section className="container-section">
      <div className="content-container text-center">
        <h2 className="text-3xl md:text-4xl font-bold mb-6 text-foreground">
          Create Topics by Goal
        </h2>
        <p className="text-xl text-muted-foreground mb-8 max-w-3xl mx-auto">
          Preparing for a standardized test? Let our AI create a comprehensive study plan tailored to your target exam. 
          Get structured topics, practice materials, and a personalized timeline to help you achieve your best score.
        </p>
        <div className="flex flex-wrap justify-center gap-6">
          {goalData.map((goal) => (
            <GoalButton
              key={goal.id}
              title={goal.title}
              description={goal.description}
              links={goal.links}
              onClick={() => onGoalSelect(goal.topic, goal.metaTopic, goal.subject)}
            />
          ))}
        </div>
      </div>
    </section>
  );
}