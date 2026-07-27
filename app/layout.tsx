import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://shinfxh.github.io"),
  title: "Reverso — Efficient Time Series Foundation Models",
  description:
    "A paper summary of Reverso, a family of compact time series foundation models for zero-shot forecasting.",
  alternates: {
    canonical: "/reverso-blog/",
  },
  openGraph: {
    type: "article",
    url: "/reverso-blog/",
    title: "Reverso — Efficient Time Series Foundation Models",
    description:
      "Competitive zero-shot forecasting with as few as 2.6 million parameters.",
    images: [{ url: "/reverso-blog/og.png", width: 1536, height: 1024 }],
  },
  twitter: {
    card: "summary_large_image",
    title: "Reverso — Efficient Time Series Foundation Models",
    description:
      "Competitive zero-shot forecasting with as few as 2.6 million parameters.",
    images: ["/reverso-blog/og.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
