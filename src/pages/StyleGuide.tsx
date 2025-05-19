import { useState } from 'react';
import '../css/index.css';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { TooltipWithContent as Tooltip } from '@/components/ui/tooltip';
import { CodeSnippet } from '@/components/CodeSnippet';

const colorTokens = [
  { name: '--color-primary', desc: 'Primary', var: 'var(--color-primary)' },
  { name: '--color-secondary', desc: 'Secondary', var: 'var(--color-secondary)' },
  { name: '--color-accent', desc: 'Accent', var: 'var(--color-accent)' },
  { name: '--color-success', desc: 'Success', var: 'var(--color-success)' },
  { name: '--color-warning', desc: 'Warning', var: 'var(--color-warning)' },
  { name: '--color-danger', desc: 'Danger', var: 'var(--color-danger)' },
  { name: '--color-background', desc: 'Background', var: 'var(--color-background)' },
  { name: '--color-surface', desc: 'Surface', var: 'var(--color-surface)' },
  { name: '--color-card', desc: 'Card', var: 'var(--color-card)' },
  { name: '--color-muted', desc: 'Muted', var: 'var(--color-muted)' },
];


const radiusTokens = [
  { name: '--radius-xs', desc: 'XS', val: 'var(--radius-xs)' },
  { name: '--radius-sm', desc: 'SM', val: 'var(--radius-sm)' },
  { name: '--radius-md', desc: 'MD', val: 'var(--radius-md)' },
  { name: '--radius-lg', desc: 'LG', val: 'var(--radius-lg)' },
  { name: '--radius-xl', desc: 'XL', val: 'var(--radius-xl)' },
];

const shadowTokens = [
  { name: '--shadow-xs', desc: 'XS', val: 'var(--shadow-xs)' },
  { name: '--shadow-sm', desc: 'SM', val: 'var(--shadow-sm)' },
  { name: '--shadow-md', desc: 'MD', val: 'var(--shadow-md)' },
  { name: '--shadow-lg', desc: 'LG', val: 'var(--shadow-lg)' },
];

const spacingTokens = [
  { name: '--space-xs', desc: 'XS', val: 'var(--space-xs)' },
  { name: '--space-sm', desc: 'SM', val: 'var(--space-sm)' },
  { name: '--space-md', desc: 'MD', val: 'var(--space-md)' },
  { name: '--space-lg', desc: 'LG', val: 'var(--space-lg)' },
  { name: '--space-xl', desc: 'XL', val: 'var(--space-xl)' },
];

const StyleGuide = () => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');

  return (
  <div className="max-w-5xl mx-auto p-8 space-y-16">
    <header className="space-y-2">
      <h1 className="text-4xl font-bold text-foreground">Style Guide</h1>
      <p className="text-muted-foreground">Design system documentation and component library</p>
    </header>

    {/* Color Tokens */}
    <section className="space-y-4">
      <div className="space-y-2">
        <h2 className="text-2xl font-semibold text-foreground">Color Tokens</h2>
        <p className="text-muted-foreground">Primary, secondary, and semantic color variables</p>
      </div>
      <div className="flex flex-wrap gap-4 p-6 bg-card rounded-lg border border-border">
        {colorTokens.map((c) => (
          <div 
            key={c.name} 
            className="w-24 h-24 component-padding rounded-lg shadow-sm flex flex-col items-center justify-center border border-border"
            style={{ background: c.var, color: c.name.includes('background') || c.name.includes('surface') || c.name.includes('card') ? 'var(--color-foreground)' : 'var(--color-background)' }}
          >
            <div className="text-xs font-semibold">{c.desc}</div>
            <div className="text-[10px] opacity-80">{c.name}</div>
          </div>
        ))}
      </div>
    </section>

    {/* Border Radius & Shadow */}
    <section className="space-y-6">
      <div className="space-y-3">
        <h2 className="text-2xl font-semibold text-foreground">Border Radius & Shadows</h2>
        <p className="text-muted-foreground">Corner radius and elevation styles</p>
      </div>
      <div className="flex flex-col md:flex-row gap-8 p-8 bg-card rounded-xl border border-border">
        <div className="space-y-4">
          <h3 className="text-lg font-medium">Border Radius</h3>
          <div className="flex flex-wrap gap-4">
            {radiusTokens.map((r) => (
              <div key={r.name} className="flex flex-col items-center">
                <div 
                  className="w-16 h-16 p-2 bg-background border border-border rounded-md flex flex-col"
                  style={{ borderRadius: r.val }}
                >
                  {r.desc}
                  <span className="text-xs text-muted-foreground mt-2">{r.name}</span>
                </div>
              </div>
            ))}
          </div>
          <p className="text-sm text-muted-foreground mt-3">Applied to: buttons, cards, inputs, etc.</p>
        </div>
        <div className="space-y-4">
          <h3 className="text-lg font-medium">Shadows</h3>
          <div className="flex flex-wrap gap-4">
            {shadowTokens.map((s) => (
              <div key={s.name} className="flex flex-col items-center">
                <div 
                  className="w-16 h-16 p-2 bg-background border border-border rounded-md flex flex-col"
                  style={{ boxShadow: s.val }}
                >
                  <span className="text-xs">{s.desc}</span>
                  <span className="text-xs text-muted-foreground mt-2">{s.name}</span>
                </div>
              </div>
            ))}
          </div>
          <p className="text-sm text-muted-foreground mt-3">Used for elevation and depth</p>
        </div>
      </div>
    </section>

    {/* Typography */}
    <section className="space-y-6">
      <div className="space-y-3">
        <h2 className="text-2xl font-semibold text-foreground">Typography</h2>
        <p className="text-muted-foreground">Font families, sizes, and text styles</p>
      </div>
      <div className="space-y-8 p-8 bg-card rounded-xl border border-border">
        <div className="space-y-2 border-b pb-6">
          <h1 className="text-4xl font-bold text-foreground">Page Title</h1>
          <p className="text-muted-foreground text-sm">Font: var(--font-sans), Size: 2.5rem, Weight: 700</p>
        </div>
        <div className="space-y-2 border-b pb-6">
          <h2 className="text-2xl font-semibold text-foreground">Section Heading</h2>
          <p className="text-muted-foreground text-sm">Font: var(--font-sans), Size: 1.5rem, Weight: 600</p>
        </div>
        <div className="space-y-2 border-b pb-6">
          <h3 className="text-xl font-semibold text-foreground">Subsection</h3>
          <p className="text-muted-foreground text-sm">Font: var(--font-sans), Size: 1.25rem, Weight: 600</p>
        </div>
        <div className="space-y-2 border-b pb-6">
          <p className="text-base text-foreground/90">Body text - The quick brown fox jumps over the lazy dog.</p>
          <p className="text-muted-foreground text-sm">Font: var(--font-sans), Size: 1rem, Weight: 400</p>
        </div>
        <div className="space-y-2">
          <p className="text-sm text-muted-foreground">Helper text and captions for additional context.</p>
          <p className="text-muted-foreground text-xs">Font: var(--font-sans), Size: 0.875rem, Weight: 400</p>
        </div>
        <div className="pt-4">
          <code className="font-mono text-sm bg-muted/50 px-3 py-2 rounded-md inline-block border border-border">
            <span className="font-semibold">Mono Example:</span> 1234567890
          </code>
          <p className="text-muted-foreground text-xs mt-2">Font: var(--font-mono), Size: 0.875rem</p>
        </div>
      </div>
    </section>

    {/* UI Components */}
    <section className="space-y-16">
      <div className="space-y-3">
        <h2 className="text-3xl font-bold text-foreground">UI Components</h2>
        <p className="text-muted-foreground">Reusable components with consistent spacing</p>
      </div>
      
      {/* Buttons */}
      <div className="space-y-6">
        <div className="space-y-3">
          <h3 className="text-2xl font-semibold text-foreground">Buttons</h3>
          <p className="text-muted-foreground">Interactive elements for user actions</p>
        </div>
        <div className="flex flex-wrap items-center gap-4 p-8 bg-card rounded-xl border border-border">
          <Button 
            variant="default"
          >
            Primary
          </Button>
          <Button 
            variant="secondary" 
          >
            Secondary
          </Button>
          <Button 
            variant="outline" 
          >
            Outline
          </Button>
          <Button 
            variant="ghost" 
          >
            Ghost
          </Button>
          <Button 
            variant="link" 
          >
            Link
          </Button>
        </div>
      </div>
      {/* Cards */}
      <div className="space-y-6">
        <div className="space-y-3">
          <h3 className="text-2xl font-semibold text-foreground">Cards</h3>
          <p className="text-muted-foreground">Containers for related content and actions</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <Card className="overflow-hidden transition-all hover:shadow-md">
            <CardHeader className="pb-3">
              <CardTitle className="text-xl">Card Title</CardTitle>
              <CardDescription>Card Description</CardDescription>
            </CardHeader>
            <CardContent className="pb-6">
              <p className="text-foreground/80">This is a card component with header, content, and footer sections.</p>
            </CardContent>
            <CardFooter className="bg-muted/20 pt-0">
              <Button variant="outline" size="sm" className="border-border/50">
                Learn More
              </Button>
            </CardFooter>
          </Card>
          <Card className="overflow-hidden transition-all hover:shadow-md">
            <CardHeader className="pb-3">
              <CardTitle className="text-xl">With Form</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-5">
                <div className="space-y-2">
                  <Label htmlFor="name" className="text-foreground/80">Name</Label>
                  <Input 
                    id="name" 
                    value={name} 
                    onChange={(e) => setName(e.target.value)}
                    className="border-border/50 focus-visible:ring-2 focus-visible:ring-primary/20"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="email" className="text-foreground/80">Email</Label>
                  <Input 
                    id="email" 
                    type="email" 
                    value={email} 
                    onChange={(e) => setEmail(e.target.value)}
                    className="border-border/50 focus-visible:ring-2 focus-visible:ring-primary/20"
                  />
                </div>
              </div>
            </CardContent>
            <CardFooter className="bg-muted/20 pt-0">
              <Button className="w-full">Save changes</Button>
            </CardFooter>
          </Card>
        </div>
      </div>

      {/* Tabs */}
      <div className="space-y-6">
        <div className="space-y-3">
          <h3 className="text-2xl font-semibold text-foreground">Tabs</h3>
          <p className="text-muted-foreground">Organize content into multiple views</p>
        </div>
        <div className="p-8 bg-card rounded-xl border border-border">
          <Tabs defaultValue="account" className="w-full max-w-2xl">
            <TabsList>
              <TabsTrigger value="account">Account</TabsTrigger>
              <TabsTrigger value="password">Password</TabsTrigger>
              <TabsTrigger value="notifications">Notifications</TabsTrigger>
            </TabsList>
            <TabsContent value="account">
              <Card>
                <CardHeader>
                  <CardTitle>Account</CardTitle>
                  <CardDescription>
                    Make changes to your account here. Click save when you're done.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="name">Name</Label>
                    <Input id="name" defaultValue="John Doe" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="email">Email</Label>
                    <Input id="email" type="email" defaultValue="john@example.com" />
                  </div>
                </CardContent>
                <CardFooter>
                  <Button>Save changes</Button>
                </CardFooter>
              </Card>
            </TabsContent>
            <TabsContent value="password">
              <Card>
                <CardHeader>
                  <CardTitle>Password</CardTitle>
                  <CardDescription>
                    Change your password here. After saving, you'll be logged out.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="current">Current password</Label>
                    <Input id="current" type="password" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="new">New password</Label>
                    <Input id="new" type="password" />
                  </div>
                </CardContent>
                <CardFooter>
                  <Button>Update password</Button>
                </CardFooter>
              </Card>
            </TabsContent>
            <TabsContent value="notifications">
              <Card>
                <CardHeader>
                  <CardTitle>Notifications</CardTitle>
                  <CardDescription>
                    Configure how you receive notifications.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between space-x-2">
                    <Label htmlFor="email-notifications" className="flex flex-col space-y-1">
                      <span>Email Notifications</span>
                      <span className="font-normal text-muted-foreground">Receive email notifications</span>
                    </Label>
                    <Input id="email-notifications" type="checkbox" className="w-5 h-5" defaultChecked />
                  </div>
                  <div className="flex items-center justify-between space-x-2">
                    <Label htmlFor="push-notifications" className="flex flex-col space-y-1">
                      <span>Push Notifications</span>
                      <span className="font-normal text-muted-foreground">Enable push notifications</span>
                    </Label>
                    <Input id="push-notifications" type="checkbox" className="w-5 h-5" />
                  </div>
                </CardContent>
                <CardFooter>
                  <Button>Save preferences</Button>
                </CardFooter>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>

      {/* Dropdown Menu */}
      <div className="space-y-4">
        <div className="space-y-2">
          <h3 className="text-2xl font-semibold text-foreground">Dropdown Menu</h3>
          <p className="text-muted-foreground">Displays a menu to the user</p>
        </div>
        <div className="p-6 bg-card rounded-lg border border-border">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline">Open Menu</Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent>
              <DropdownMenuItem>Profile</DropdownMenuItem>
              <DropdownMenuItem>Settings</DropdownMenuItem>
              <DropdownMenuItem>Logout</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* Tooltip */}
      <div className="space-y-4">
        <div className="space-y-2">
          <h3 className="text-2xl font-semibold text-foreground">Tooltip</h3>
          <p className="text-muted-foreground">Contextual information on hover</p>
        </div>
        <div className="p-6 bg-card rounded-lg border border-border">
          <Tooltip content="This is a tooltip">
            <Button variant="outline">Hover me</Button>
          </Tooltip>
        </div>
      </div>
    </section>

    {/* Code Block Example */}
    <section className="space-y-6">
      <div className="space-y-3">
        <h2 className="text-2xl font-semibold text-foreground">Code Examples</h2>
        <p className="text-muted-foreground">Syntax highlighting and code formatting</p>
      </div>
      <div className="space-y-8">
        <div className="space-y-4">
          <h3 className="text-lg font-medium text-foreground">JavaScript Example</h3>
          <div className="p-0 bg-card rounded-xl border border-border overflow-hidden">
            <CodeSnippet 
              language="javascript"
              code={
`// Function definition
function greet(name) {
  return 'Hello, ' + name + '!';
}

// Example usage
const message = greet('World');
console.log(message); // Output: Hello, World!`
              }
            />
          </div>
        </div>
      </div>
    </section>

    {/* Utility Classes */}
    <section className="space-y-8">
      <div className="space-y-3">
        <h2 className="text-2xl font-semibold text-foreground">Utility Classes</h2>
        <p className="text-muted-foreground">Helper classes for common styling needs</p>
      </div>
      <div className="space-y-8">
        <div className="space-y-4">
          <h3 className="text-lg font-medium text-foreground">Colors</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 p-8 bg-card rounded-xl border border-border">
            <div className="flex flex-col items-center p-4 rounded bg-primary text-primary-foreground">
              <span className="font-medium">Primary</span>
              <span className="text-xs opacity-80">.bg-primary</span>
            </div>
            <div className="flex flex-col items-center p-4 rounded bg-secondary text-secondary-foreground">
              <span className="font-medium">Secondary</span>
              <span className="text-xs opacity-80">.bg-secondary</span>
            </div>
            <div className="flex flex-col items-center p-4 rounded bg-accent text-accent-foreground">
              <span className="font-medium">Accent</span>
              <span className="text-xs opacity-80">.bg-accent</span>
            </div>
            <div className="flex flex-col items-center p-4 rounded bg-destructive text-destructive-foreground">
              <span className="font-medium">Destructive</span>
              <span className="text-xs opacity-80">.bg-destructive</span>
            </div>
          </div>
        </div>
        
        <div className="space-y-4">
          <h3 className="text-lg font-medium text-foreground">Text Colors</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 p-8 bg-card rounded-xl border border-border">
            <div className="flex flex-col items-center p-4 rounded border">
              <span className="text-foreground font-medium">Foreground</span>
              <span className="text-xs text-muted-foreground">.text-foreground</span>
            </div>
            <div className="flex flex-col items-center p-4 rounded border">
              <span className="text-muted-foreground font-medium">Muted</span>
              <span className="text-xs text-muted-foreground">.text-muted-foreground</span>
            </div>
            <div className="flex flex-col items-center p-4 rounded border">
              <span className="text-primary font-medium">Primary</span>
              <span className="text-xs text-muted-foreground">.text-primary</span>
            </div>
            <div className="flex flex-col items-center p-4 rounded border">
              <span className="text-destructive font-medium">Destructive</span>
              <span className="text-xs text-muted-foreground">.text-destructive</span>
            </div>
          </div>
        </div>
        
      </div>
    </section>
  </div>
  );
};

export default StyleGuide;
