import { PageHeading } from '../components/common';
import { IdeaTree } from '../components/IdeaTree';

export function IdeaTreePage() {
  return <>
    <PageHeading
      eyebrow="Research map"
      title="Idea tree"
      description="Every main idea the project tested, in the order it happened, coloured by whether it worked. Select a box for a short summary, then open that test’s page."
    />
    <IdeaTree/>
  </>;
}
