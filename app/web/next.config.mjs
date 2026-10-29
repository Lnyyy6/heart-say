const API_PROXY = process.env.API_PROXY_TARGET || "http://localhost:8010";

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${API_PROXY}/:path*`,
      },
    ];
  },
};

export default nextConfig;
