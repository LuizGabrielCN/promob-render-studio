import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cn } from "@/lib/utils";

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  asChild?: boolean;
  variant?: "default" | "destructive" | "outline" | "secondary" | "ghost" | "link";
  size?: "default" | "sm" | "lg" | "icon";
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", size = "default", asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";

    const baseClasses =
      "inline-flex items-center justify-center whitespace-nowrap rounded-md text-xs font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 select-none";

    const variantClasses = {
      default: "bg-zinc-100 text-zinc-900 hover:bg-zinc-200 active:bg-zinc-300 font-semibold",
      destructive: "bg-red-900/60 text-red-200 border border-red-800 hover:bg-red-900 active:bg-red-950",
      outline: "border border-zinc-800 bg-transparent hover:bg-zinc-900 hover:text-zinc-100 active:bg-zinc-800/80",
      secondary: "bg-zinc-800/80 text-zinc-200 hover:bg-zinc-800 active:bg-zinc-700/80 border border-zinc-750",
      ghost: "hover:bg-zinc-900 hover:text-zinc-100 text-zinc-400 active:bg-zinc-800",
      link: "text-zinc-400 underline-offset-4 hover:underline",
    };

    const sizeClasses = {
      default: "h-9 px-4 py-2",
      sm: "h-8 rounded-md px-3 text-xs",
      lg: "h-10 rounded-md px-6 text-sm",
      icon: "h-8 w-8",
    };

    return (
      <Comp
        className={cn(baseClasses, variantClasses[variant], sizeClasses[size], className)}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export { Button };
