/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: true,
    // Explicitly set distDir to prevent WSL/Windows path issues
    distDir: '.next',
};

module.exports = nextConfig;

