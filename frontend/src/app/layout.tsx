import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ClauseGuard — AI Contract Risk Audit",
  description:
    "Automated legal contract review powered by LegalBERT classification, NuExtract 2.0 extraction, and deterministic policy rules.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      </head>
      <body>{children}</body>
    </html>
  );
}
