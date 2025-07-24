import * as React from "react"
import {
  DropdownMenu as ShadcnDropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Button } from "@/components/ui/button"
import { ChevronDown } from "lucide-react"

interface DropdownMenuProps {
  triggerLabel: string
  items: {
    label: string
    onClick: () => void
    disabled?: boolean
  }[]
  align?: "start" | "end" | "center"
}

export function DropdownMenu({ triggerLabel, items, align = "end" }: DropdownMenuProps) {
  return (
    <ShadcnDropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" className="flex items-center gap-2">
          {triggerLabel}
          <ChevronDown className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align={align}>
        {items.map((item, index) => (
          <React.Fragment key={item.label}>
            <DropdownMenuItem
              onClick={item.onClick}
              disabled={item.disabled}
            >
              {item.label}
            </DropdownMenuItem>
            {index < items.length - 1 && <DropdownMenuSeparator />}
          </React.Fragment>
        ))}
      </DropdownMenuContent>
    </ShadcnDropdownMenu>
  )
} 