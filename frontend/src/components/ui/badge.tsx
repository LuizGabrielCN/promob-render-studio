import * as React from "react";
import { cn } from "@/lib/utils";

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "secondary" | "outline" | "success" | "warning" | "destructive";
}

function Badge({ className, variant = "default", ...props }: BadgeProps) {
  const variantClasses = {
    default: "border-transparent bg-zinc-800 text-zinc-200",
    secondary: "border-zinc-800 bg-zinc-900/80 text-zinc-400",
    outline: "text-zinc-300 border-zinc-800",
    success: "border-emerald-800/60 bg-emerald-950/40 text-emerald-400",
    warning: "border-amber-800/60 bg-amber-950/40 text-amber-300",
    destructive: "border-red-800/60 bg-red-950/40 text-red-300",
  };

  return (
    <div
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] font-medium tracking-tight transition-colors",
        variantClasses[variant],
        className
      )}
      {...props}
    />
  );
}

export { Badge };
