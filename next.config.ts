import type { NextConfig } from "next";

const isGitHubPages = process.env.GITHUB_PAGES === "true";

const nextConfig: NextConfig = isGitHubPages
  ? {
      output: "export",
      basePath: "/reverso-blog",
      assetPrefix: "/reverso-blog/",
      images: { unoptimized: true },
      trailingSlash: true,
    }
  : {};

export default nextConfig;
