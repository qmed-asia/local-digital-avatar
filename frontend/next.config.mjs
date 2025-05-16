/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: false,
    async rewrites() {
        return [
           
            {
                source: '/api/stt/v1/:slug*',
                destination: `http://${process.env.NEXT_PUBLIC_STT_URL}/v1/:slug*`,
            },

        ]
    }
};

export default nextConfig;
