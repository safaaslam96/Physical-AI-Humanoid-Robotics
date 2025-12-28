import { betterAuth } from "better-auth";

const auth = betterAuth({
  database: {
    provider: "sqlite",
    url: process.env.DATABASE_URL || "./db.sqlite",
  },
  secret: process.env.BETTER_AUTH_SECRET || "your-super-secret-key-change-in-production",
  emailAndPassword: {
    enabled: true,
    requireEmailVerification: false,
  },
  socialProviders: {
    google: {
      clientId: process.env.GOOGLE_CLIENT_ID || "",
      clientSecret: process.env.GOOGLE_CLIENT_SECRET || "",
    },
  },
  user: {
    // Custom fields for user profile (cast to any to allow additional keys not declared by the library types)
    fields: {
      // Custom fields for user profile
      softwareBackground: "software_background",
      hasHighEndGPU: "has_high_end_gpu",
      familiarWithROS2: "familiar_with_ros2",
    } as any,
  },
  plugins: [
    // Add any additional plugins here
  ],
});

export default auth;